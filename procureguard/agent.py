"""
ProcureGuard Agent — tool-calling loop powered by Distil Labs SLM.

The agent:
1. Recalls memory from cognee
2. Asks the SLM to select tools
3. Executes tools against Qdrant / cognee
4. Asks the SLM to synthesize the answer
5. Yields SSE events at every step ("think out loud")
"""

import json
import re
import os
import sys
import asyncio
import time
from typing import AsyncGenerator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.llm import get_llm_response, get_model_name

from tools import (
    get_tool_descriptions_text,
    execute_tool,
    recall_memory,
    store_finding,
)

# ── prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "system_prompt.txt")

def _load_system_prompt() -> str:
    try:
        with open(SYSTEM_PROMPT_PATH) as f:
            return f.read()
    except FileNotFoundError:
        return ""

TOOL_SELECTION_PROMPT = """You are ProcureGuard, an AI procurement auditor with persistent memory and access to tools.

Available tools:
{tools}

Given the user question and any memory context, decide which tool(s) to call.
Respond with one or more tool calls in EXACTLY this format (one per tool):

TOOL: tool_name
PARAMS: {{"param": "value"}}

If you need multiple tools, separate them with a line containing only ---.
If no tool is needed, respond with TOOL: none

RULES:
- For payment verification questions: use search_knowledge AND detect_anomalies
- For spend comparison questions: use analyze_spend
- For pricing, cost comparison, overpriced, or cheapest vendor questions: use analyze_pricing
- For "suspicious" or "anomaly" questions: use detect_anomalies
- For investigation of specific records: use investigate_record
- For questions about past findings: use recall_memory
- Always be specific with parameters
"""

SYNTHESIS_PROMPT_TEMPLATE = """You are ProcureGuard, an AI procurement auditor.

{system_prompt}

You have just executed these tools and received results:

{tool_results}

Memory context from past investigations:
{memory_context}

User question: {question}

Provide a clear, specific answer. ALWAYS:
- Cite specific transaction IDs, invoice numbers, amounts, dates, SKUs
- INFER patterns when asked about discrepancies, consistency, or trends
- Never say "the data contains no information about X" if X can be deduced from patterns
- For comparisons, provide exact dollar amounts
- Be concise: 2-5 sentences for simple questions, more for investigations
"""

# ── conversation history ─────────────────────────────────────────────────────

_conversation_history: list[dict] = []


def get_conversation_history() -> list[dict]:
    return list(_conversation_history)


def _add_to_history(role: str, content: str):
    _conversation_history.append({"role": role, "content": content[:500]})
    if len(_conversation_history) > 20:
        _conversation_history.pop(0)


# ── tool-call parser ─────────────────────────────────────────────────────────

def parse_tool_calls(response: str) -> list[tuple[str, dict]]:
    """Parse SLM response into list of (tool_name, params) tuples."""
    calls = []
    # Split on --- separator
    blocks = re.split(r'\n-{3,}\n', response)
    for block in blocks:
        tool_match = re.search(r'TOOL:\s*(\w+)', block)
        params_match = re.search(r'PARAMS:\s*(\{.*?\})', block, re.DOTALL)
        if tool_match:
            name = tool_match.group(1).strip()
            if name == "none":
                continue
            params = {}
            if params_match:
                try:
                    params = json.loads(params_match.group(1))
                except json.JSONDecodeError:
                    pass
            calls.append((name, params))
    # Fallback: if no tools parsed, try to do a simple search
    if not calls:
        calls.append(("search_knowledge", {"query": response[:200]}))
    return calls


# ── SSE event helper ─────────────────────────────────────────────────────────

def _sse_event(step: str, **kwargs) -> str:
    """Create a JSON SSE event string."""
    return json.dumps({"step": step, **kwargs})


# ── main agent stream ────────────────────────────────────────────────────────

async def agent_stream(user_query: str) -> AsyncGenerator[str, None]:
    """
    Stream the full agent reasoning loop as SSE events.
    Each yield is a JSON-encoded event for the frontend.
    """
    _add_to_history("user", user_query)

    # ── Step 1: Recall memory ────────────────────────────────────────────
    yield _sse_event("thinking", content="Checking knowledge graph for relevant memory...")

    memory_result = await recall_memory(user_query)
    memory_context = "\n".join(memory_result.get("results", []))

    if memory_context and "error" not in memory_context.lower():
        yield _sse_event("memory", content=f"Found {memory_result['count']} memory fragments", results=memory_result["results"][:3])
    else:
        yield _sse_event("memory", content="No prior findings in memory")
        memory_context = "No previous findings."

    # ── Step 2: Tool selection via SLM ───────────────────────────────────
    yield _sse_event("thinking", content="Selecting tools...")

    history_text = ""
    if _conversation_history:
        recent = _conversation_history[-6:]
        history_text = "\n".join(f"{m['role']}: {m['content']}" for m in recent)

    selection_prompt = f"""Memory context: {memory_context[:500]}

Recent conversation:
{history_text}

User question: {user_query}

Select the tools needed to answer this question."""

    tool_selection_system = TOOL_SELECTION_PROMPT.format(tools=get_tool_descriptions_text())
    selection_response = get_llm_response(tool_selection_system, selection_prompt, max_tokens=512)

    tool_calls = parse_tool_calls(selection_response)
    tool_names = [t[0] for t in tool_calls]

    yield _sse_event("tools_selected", tools=tool_names, raw=selection_response[:300])

    # ── Step 3: Execute tools ────────────────────────────────────────────
    all_tool_results = {}

    for tool_name, params in tool_calls:
        yield _sse_event("tool_call", tool=tool_name, params=params)

        result = await execute_tool(tool_name, params)
        all_tool_results[tool_name] = result

        # Build a brief summary for the UI
        summary = _summarize_tool_result(tool_name, result)
        yield _sse_event("tool_result", tool=tool_name, summary=summary)

    # ── Step 4: Synthesize answer via SLM ────────────────────────────────
    yield _sse_event("thinking", content="Synthesizing answer from tool results...")

    base_system = _load_system_prompt()
    tool_results_text = json.dumps(all_tool_results, indent=2, default=str)[:6000]

    synthesis_system = SYNTHESIS_PROMPT_TEMPLATE.format(
        system_prompt=base_system[:2000],
        tool_results=tool_results_text,
        memory_context=memory_context[:500],
        question=user_query,
    )

    answer = get_llm_response(
        "You are ProcureGuard, an AI procurement auditor. Be specific with IDs, amounts, and dates.",
        synthesis_system,
        max_tokens=1024,
    )

    _add_to_history("assistant", answer)

    yield _sse_event("answer", content=answer, model=get_model_name(), tools_used=tool_names)

    # ── Step 5: Store finding if significant ─────────────────────────────
    significant_keywords = ["anomal", "discrepanc", "outlier", "suspicious", "flag", "risk", "duplicate", "finding"]
    if any(kw in answer.lower() for kw in significant_keywords):
        yield _sse_event("thinking", content="Storing finding in knowledge graph...")
        finding_text = f"AUDIT FINDING from ProcureGuard:\nQuestion: {user_query}\nAnswer: {answer[:500]}"
        store_result = await store_finding(finding_text)
        yield _sse_event("memory_stored", status=store_result.get("status", "unknown"))

    yield _sse_event("done")


def _summarize_tool_result(tool_name: str, result: dict) -> str:
    """Create a short human-readable summary of a tool result."""
    if tool_name == "search_knowledge":
        n = result.get("count", 0)
        return f"Found {n} matching records ({result.get('search_ms', '?')}ms)"
    elif tool_name == "analyze_spend":
        total = result.get("total_spend", 0)
        return f"Total spend: ${total:,.0f}" if total else "Analytics loaded"
    elif tool_name == "detect_anomalies":
        n = result.get("count", 0)
        high = sum(1 for a in result.get("anomalies", []) if a.get("severity") == "high")
        return f"Found {n} anomalies ({high} high severity)"
    elif tool_name == "investigate_record":
        n = result.get("count", 0)
        return f"Found {n} similar records ({result.get('time_ms', '?')}ms)"
    elif tool_name == "recall_memory":
        n = result.get("count", 0)
        return f"Recalled {n} memory fragments"
    elif tool_name == "store_finding":
        return f"Finding stored: {result.get('status', 'unknown')}"
    elif tool_name == "ingest_document":
        return f"Document ingested: {result.get('status', 'unknown')}"
    elif tool_name == "analyze_pricing":
        n = result.get("products_analyzed", 0)
        savings = result.get("total_savings_potential", 0)
        return f"Analyzed {n} products, potential savings: ${savings:,.0f}"
    return json.dumps(result, default=str)[:200]
