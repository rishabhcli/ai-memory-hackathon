"""
ProcureGuard — AI Procurement Auditor with Structured Memory

Unified FastAPI app combining:
- Cognee knowledge graph memory (entity extraction, relationship mapping)
- Qdrant vector search (7 advanced APIs)
- Distil Labs SLM reasoning (tool selection, synthesis, inference)
- Live knowledge graph visualization (D3.js)
- Adversarial fraud simulation (Red Team mode)

Port: 8000
"""

# ── CRITICAL: Set cognee env vars BEFORE any cognee imports ─────────────────
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)

os.environ.setdefault("VECTOR_DB_PROVIDER", "qdrant")
os.environ.setdefault("VECTOR_DB_URL", os.getenv("QDRANT_URL", "http://localhost:6333"))
os.environ.setdefault("VECTOR_DB_KEY", os.getenv("QDRANT_API_KEY", ""))
os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")
os.environ.setdefault("LLM_API_KEY", os.getenv("LLM_API_KEY", "."))

# Register Qdrant adapter BEFORE importing cognee
import cognee_community_vector_adapter_qdrant.register  # noqa: F401, E402

import cognee  # noqa: E402
from cognee.api.v1.search import SearchType  # noqa: E402

# ── Standard imports ────────────────────────────────────────────────────────
import json  # noqa: E402
import time  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI, Query, Request  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse  # noqa: E402
from sse_starlette.sse import EventSourceResponse  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import PayloadSchemaType  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.llm import init_llm, get_llm_response, get_model_name  # noqa: E402
from shared.embeddings import init_embeddings, get_embedding  # noqa: E402

import tools  # noqa: E402
import agent  # noqa: E402
from graph_viz import extract_subgraph  # noqa: E402

# ── Paths ───────────────────────────────────────────────────────────────────
EMBED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "nomic-embed-text", "nomic-embed-text-v1.5.f16.gguf")
LLM_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "cognee-distillabs-model-gguf-quantized", "model-quantized.gguf")
LLM_FALLBACK_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen3-4B-Q4_K_M", "Qwen3-4B-Q4_K_M.gguf")

# ── Qdrant client ───────────────────────────────────────────────────────────
qdrant = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"), api_key=os.environ.get("QDRANT_API_KEY") or None)

# ── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("  ProcureGuard — Starting up")
    print("=" * 60)

    # Init shared embedding + LLM modules
    init_embeddings(EMBED_MODEL_PATH)
    init_llm([(LLM_MODEL_PATH, "Distil Labs"), (LLM_FALLBACK_PATH, "Qwen3-4B")])

    # Init tools with Qdrant client
    tools.init_tools(qdrant)

    # Payload indexes
    for col in ["DocumentChunk_text", "Entity_name"]:
        for field, schema in [("type", PayloadSchemaType.KEYWORD), ("text", PayloadSchemaType.TEXT)]:
            try:
                qdrant.create_payload_index(collection_name=col, field_name=field, field_schema=schema)
            except Exception:
                pass

    # Load data + compute analytics + anomalies
    print("Loading records from Qdrant...")
    records = tools.load_all_records()
    print(f"  Loaded {len(records)} records")

    print("Computing analytics...")
    analytics = tools.compute_analytics()
    print(f"  Total spend: ${analytics['total_spend']:,.0f}")

    print("Running anomaly detection...")
    anomalies = tools.run_anomaly_detection()
    print(f"  Found {anomalies['summary']['total']} anomalies ({anomalies['summary']['high']} high)")

    print("Computing pricing intelligence...")
    pricing = tools.compute_pricing_intelligence()
    print(f"  Analyzed {pricing['products_analyzed']} products, savings potential: ${pricing['total_savings_potential']:,.0f}")

    print("=" * 60)
    print("  ProcureGuard — Ready on http://localhost:8000")
    print("=" * 60)
    yield


app = FastAPI(title="ProcureGuard", lifespan=lifespan)


# ── API Routes ──────────────────────────────────────────────────────────────

@app.get("/api/ask")
async def api_ask(q: str = Query(...)):
    """Agent Q&A with SSE streaming of reasoning steps."""
    return EventSourceResponse(agent.agent_stream(q))


@app.get("/api/analytics")
async def api_analytics():
    """Return cached analytics."""
    return tools._analytics_cache


@app.get("/api/anomalies")
async def api_anomalies(vendor_id: int | None = None, anomaly_type: str | None = None):
    """Return cached anomalies, optionally filtered."""
    result = await tools.detect_anomalies(vendor_id=vendor_id, anomaly_type=anomaly_type)
    return result


@app.get("/api/graph")
async def api_graph(q: str = Query(...)):
    """Extract a focused subgraph for D3 visualization."""
    graph = extract_subgraph(qdrant, q, get_embedding, limit=15)
    return graph


@app.get("/api/search")
async def api_search(q: str = Query(...), collection: str = "DocumentChunk_text", limit: int = 10):
    """Direct Qdrant search (Prefetch + RRF Fusion)."""
    result = await tools.search_knowledge(q, collection, limit)
    return result


@app.post("/api/ingest")
async def api_ingest(request: Request):
    """Ingest raw text through cognee pipeline."""
    body = await request.json()
    raw_text = body.get("text", "")
    if not raw_text:
        return JSONResponse({"error": "No text provided"}, status_code=400)
    result = await tools.ingest_document(raw_text)
    return result


@app.get("/api/cognee-search")
async def api_cognee_search(q: str = Query(...), search_type: str = "CHUNKS"):
    """Search via cognee with specified SearchType."""
    t0 = time.time()
    try:
        st = getattr(SearchType, search_type.upper(), SearchType.CHUNKS)
        results = await cognee.search(query_text=q, query_type=st)
        items = [str(r)[:300] for r in (results or [])[:10]]
    except Exception as e:
        return {"error": str(e)}
    return {"query": q, "search_type": search_type, "results": items, "count": len(items), "time_ms": round((time.time() - t0) * 1000, 1)}


@app.post("/api/redteam/generate")
async def api_redteam_generate():
    """Generate a synthetic fraudulent invoice via SLM."""
    prompt = """Generate a single realistic procurement invoice that contains ONE subtle fraud indicator.
Choose one fraud type: slightly inflated price (5-15% above normal), duplicate line item, or unusual quantity.
Use vendors 1-20, products from: monitors, laptops, SSDs, RAM, keyboards, HDDs.
Use this EXACT format:
Invoice INV-V{vendor_id}-M01-{6_digit_random} from Vendor {vendor_id}, date 2025-06-15, total ${total}
{qty}x {product_name} (SKU: {sku}) at ${unit_price} = ${line_total}

After the invoice, on a new line write:
FRAUD: {one sentence describing the hidden fraud indicator}"""

    invoice_text = get_llm_response(
        "You are a procurement data generator for testing fraud detection systems. Generate realistic but subtly fraudulent invoices.",
        prompt,
        max_tokens=400,
    )

    # Split out the fraud indicator
    fraud_hint = ""
    lines = invoice_text.strip().split("\n")
    clean_lines = []
    for line in lines:
        if line.strip().upper().startswith("FRAUD:"):
            fraud_hint = line.strip()[6:].strip()
        else:
            clean_lines.append(line)

    return {"invoice": "\n".join(clean_lines), "fraud_hint": fraud_hint, "model": get_model_name()}


@app.post("/api/redteam/detect")
async def api_redteam_detect(request: Request):
    """Ingest a synthetic invoice and try to detect fraud."""
    body = await request.json()
    invoice_text = body.get("invoice", "")
    if not invoice_text:
        return JSONResponse({"error": "No invoice text"}, status_code=400)

    steps = []

    # Step 1: Ingest through cognee
    t0 = time.time()
    try:
        await cognee.add(invoice_text)
        await cognee.cognify()
        steps.append({"step": "ingest", "status": "success", "time_ms": round((time.time() - t0) * 1000, 1)})
    except Exception as e:
        steps.append({"step": "ingest", "status": f"error: {e}"})

    # Step 2: Search for similar records
    t1 = time.time()
    search_result = await tools.search_knowledge(invoice_text[:200], limit=5)
    similar = search_result.get("results", [])
    steps.append({"step": "search_similar", "count": len(similar), "time_ms": round((time.time() - t1) * 1000, 1)})

    # Step 3: Ask SLM to analyze
    context = json.dumps(similar[:3], default=str)[:3000]
    analysis = get_llm_response(
        "You are a fraud detection system. Compare the new invoice against similar existing records. Look for: price discrepancies, unusual quantities, duplicate patterns, mismatched SKUs.",
        f"NEW INVOICE:\n{invoice_text}\n\nSIMILAR EXISTING RECORDS:\n{context}\n\nAnalysis: Is this invoice fraudulent? What specific indicator did you find?",
        max_tokens=400,
    )

    detected = any(w in analysis.lower() for w in ["fraud", "suspicious", "anomal", "inflat", "discrepanc", "unusual", "duplicate"])
    steps.append({"step": "analysis", "detected": detected, "explanation": analysis})

    return {"detected": detected, "steps": steps, "explanation": analysis, "model": get_model_name()}


@app.get("/api/history")
async def api_history():
    """Return conversation history."""
    return {"history": agent.get_conversation_history()}


@app.get("/api/pricing")
async def api_pricing():
    """Return cached pricing intelligence data."""
    return tools._pricing_cache


@app.post("/api/pricing/recommend")
async def api_pricing_recommend():
    """Generate SLM-powered pricing optimization recommendations."""
    pricing = tools._pricing_cache
    overpriced = pricing.get("overpriced_products", [])[:8]
    summary = json.dumps({
        "total_savings_potential": pricing.get("total_savings_potential", 0),
        "products_analyzed": pricing.get("products_analyzed", 0),
        "overpriced": overpriced,
        "top_products": {k: {"best_vendor": v["best_vendor"], "best_price": v["best_price"],
                             "worst_price": v["worst_price"], "savings": v["savings_potential"]}
                         for k, v in list(pricing.get("product_vendor_prices", {}).items())[:10]},
    }, indent=2, default=str)

    raw = get_llm_response(
        "You are a procurement pricing optimization expert. Analyze the pricing data and provide 4-6 specific, actionable recommendations. For each, state the product, the action (switch vendor / negotiate / consolidate orders), and the estimated savings.",
        f"Pricing data:\n{summary}\n\nProvide recommendations in this format:\nRECO: product | action | detail | savings_amount",
        max_tokens=600,
    )

    # Parse recommendations
    recos = []
    for line in raw.split("\n"):
        line = line.strip()
        if line.upper().startswith("RECO:"):
            parts = line[5:].split("|")
            if len(parts) >= 3:
                recos.append({
                    "product": parts[0].strip(),
                    "action": parts[1].strip(),
                    "detail": parts[2].strip(),
                    "savings": parts[3].strip() if len(parts) > 3 else "",
                })

    return {"recommendations": recos if recos else None, "raw": raw, "model": get_model_name()}


# ── Static file serving ─────────────────────────────────────────────────────

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
async def landing_page():
    """Serve the landing page."""
    return FileResponse(STATIC_DIR / "landing.html")


@app.get("/app", response_class=HTMLResponse)
async def app_page():
    """Serve the main application."""
    return FileResponse(STATIC_DIR / "app.html")


# ── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
