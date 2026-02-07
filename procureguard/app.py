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
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402
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


# ── Main HTML UI ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_UI


HTML_UI = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ProcureGuard</title>
<script src="https://d3js.org/d3.v5.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui,-apple-system,sans-serif;background:#0a0a0a;color:#e0e0e0;height:100vh;overflow:hidden}
.header{background:#111;border-bottom:1px solid #222;padding:0.75rem 1.5rem;display:flex;align-items:center;gap:1rem;justify-content:space-between}
.header h1{font-size:1.3rem;color:#7c3aed}
.header .badges{display:flex;gap:0.5rem}
.badge{padding:2px 8px;border-radius:4px;font-size:0.7rem}
.badge.cognee{background:#1e1b4b;color:#a5b4fc}.badge.qdrant{background:#0c4a6e;color:#7dd3fc}.badge.distil{background:#065f46;color:#6ee7b7}.badge.doo{background:#78350f;color:#fcd34d}
.main{display:grid;grid-template-columns:1fr 360px;height:calc(100vh - 52px)}
.chat-panel{display:flex;flex-direction:column;border-right:1px solid #222}
.messages{flex:1;overflow-y:auto;padding:1rem;display:flex;flex-direction:column;gap:0.75rem}
.msg{max-width:90%;padding:0.75rem 1rem;border-radius:12px;font-size:0.9rem;line-height:1.5}
.msg.user{align-self:flex-end;background:#1e1b4b;color:#c4b5fd;border-bottom-right-radius:4px}
.msg.agent{align-self:flex-start;background:#1a1a1a;border:1px solid #333;border-bottom-left-radius:4px}
.msg.agent .answer{color:#e0e0e0}
.reasoning{margin-bottom:0.5rem}
.reasoning .step{display:flex;align-items:flex-start;gap:0.5rem;padding:0.25rem 0;font-size:0.8rem;color:#888}
.reasoning .step .icon{flex-shrink:0;width:18px;text-align:center}
.reasoning .step.highlight{color:#7c3aed}
.reasoning .step.error{color:#ef4444}
.input-row{display:flex;gap:0.5rem;padding:0.75rem 1rem;border-top:1px solid #222;background:#111}
.input-row input{flex:1;padding:0.75rem;border-radius:8px;border:1px solid #333;background:#1a1a1a;color:#fff;font-size:0.95rem;outline:none}
.input-row input:focus{border-color:#7c3aed}
.input-row button{padding:0.75rem 1.25rem;border-radius:8px;border:none;background:#7c3aed;color:#fff;cursor:pointer;font-size:0.95rem;font-weight:600}
.input-row button:hover{background:#6d28d9}
.input-row button:disabled{opacity:0.5;cursor:not-allowed}
.sidebar{display:flex;flex-direction:column;overflow-y:auto;background:#0f0f0f}
.sidebar-section{padding:1rem;border-bottom:1px solid #222}
.sidebar-section h3{font-size:0.85rem;color:#888;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.75rem}
.kpi-grid{display:grid;grid-template-columns:1fr 1fr;gap:0.5rem}
.kpi{background:#1a1a1a;border:1px solid #222;border-radius:8px;padding:0.6rem;text-align:center}
.kpi .val{font-size:1.3rem;font-weight:700;color:#7c3aed}
.kpi .val.amber{color:#f59e0b}.kpi .val.red{color:#ef4444}.kpi .val.green{color:#22c55e}
.kpi .lbl{font-size:0.7rem;color:#666;margin-top:0.15rem}
#graph-container{width:100%;height:250px;background:#111;border-radius:8px;overflow:hidden;position:relative}
#graph-container svg{width:100%;height:100%}
#graph-placeholder{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#444;font-size:0.85rem}
.actions{display:flex;gap:0.5rem;flex-wrap:wrap}
.actions button{padding:0.5rem 0.75rem;border-radius:6px;border:1px solid #333;background:#1a1a1a;color:#ccc;cursor:pointer;font-size:0.8rem}
.actions button:hover{border-color:#7c3aed;color:#fff}
.actions button.red{border-color:#991b1b;color:#fca5a5}.actions button.red:hover{background:#991b1b;color:#fff}
.anomaly-list{max-height:200px;overflow-y:auto}
.anomaly-item{padding:0.4rem 0;border-bottom:1px solid #1a1a1a;font-size:0.8rem;display:flex;gap:0.5rem;align-items:center}
.sev{padding:1px 6px;border-radius:3px;font-size:0.7rem;font-weight:600}
.sev.high{background:#991b1b;color:#fca5a5}.sev.medium{background:#78350f;color:#fcd34d}
.memory-count{color:#a5b4fc;font-size:0.85rem}
.modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.7);z-index:100;align-items:center;justify-content:center}
.modal-overlay.show{display:flex}
.modal{background:#1a1a1a;border:1px solid #333;border-radius:12px;padding:1.5rem;width:500px;max-width:90vw}
.modal h3{color:#7c3aed;margin-bottom:1rem}
.modal textarea{width:100%;height:150px;background:#111;border:1px solid #333;border-radius:8px;color:#fff;padding:0.75rem;font-family:monospace;font-size:0.85rem;resize:vertical}
.modal .btn-row{display:flex;gap:0.5rem;margin-top:1rem;justify-content:flex-end}
.modal button{padding:0.5rem 1rem;border-radius:6px;border:none;cursor:pointer;font-size:0.85rem}
.modal button.primary{background:#7c3aed;color:#fff}.modal button.secondary{background:#333;color:#ccc}
</style></head><body>

<div class="header">
  <h1>ProcureGuard</h1>
  <div class="badges">
    <span class="badge cognee">cognee Memory</span>
    <span class="badge qdrant">Qdrant Search</span>
    <span class="badge distil">Distil Labs SLM</span>
    <span class="badge doo">DigitalOcean</span>
  </div>
</div>

<div class="main">
  <div class="chat-panel">
    <div class="messages" id="messages">
      <div class="msg agent">
        <div class="answer">Welcome to <strong>ProcureGuard</strong>. I'm your AI procurement auditor with persistent memory.<br><br>
        I have access to 1,000 invoices and 1,000 transactions across 20 vendors. Ask me anything about payments, vendors, products, or suspicious activity.<br><br>
        <em>Try: "Check whether all payments to Vendor 2 are correct"</em></div>
      </div>
    </div>
    <div class="input-row">
      <input id="userInput" placeholder="Ask ProcureGuard..." autofocus />
      <button id="sendBtn" onclick="sendQuery()">Ask</button>
    </div>
  </div>

  <div class="sidebar">
    <div class="sidebar-section">
      <h3>Dashboard</h3>
      <div class="kpi-grid">
        <div class="kpi"><div class="val" id="kpi-spend">-</div><div class="lbl">Total Spend</div></div>
        <div class="kpi"><div class="val green" id="kpi-invoices">-</div><div class="lbl">Invoices</div></div>
        <div class="kpi"><div class="val red" id="kpi-anomalies">-</div><div class="lbl">Anomalies</div></div>
        <div class="kpi"><div class="val amber" id="kpi-vendors">-</div><div class="lbl">Vendors</div></div>
      </div>
    </div>

    <div class="sidebar-section">
      <h3>Knowledge Graph</h3>
      <div id="graph-container"><div id="graph-placeholder">Ask a question to explore the graph</div><svg></svg></div>
    </div>

    <div class="sidebar-section">
      <h3>Actions</h3>
      <div class="actions">
        <button class="red" onclick="startRedTeam()">Red Team</button>
        <button onclick="showIngestModal()">Add Document</button>
      </div>
    </div>

    <div class="sidebar-section">
      <h3>Top Anomalies</h3>
      <div class="anomaly-list" id="anomaly-list">Loading...</div>
    </div>

    <div class="sidebar-section">
      <h3>Memory</h3>
      <div class="memory-count" id="memory-count">Checking...</div>
    </div>
  </div>
</div>

<!-- Ingest Modal -->
<div class="modal-overlay" id="ingestModal">
  <div class="modal">
    <h3>Add Procurement Document</h3>
    <p style="color:#888;font-size:0.85rem;margin-bottom:0.75rem">Paste raw invoice or transaction text. It will be processed through cognee's entity extraction pipeline.</p>
    <textarea id="ingestText" placeholder="Invoice INV-V9-M03-779589 from Vendor 9, total $25,325.84..."></textarea>
    <div class="btn-row">
      <button class="secondary" onclick="closeIngestModal()">Cancel</button>
      <button class="primary" onclick="submitIngest()">Ingest</button>
    </div>
  </div>
</div>

<script>
const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
let isStreaming = false;

// ── Load dashboard data ────────────────────────────────────────────
fetch('/api/analytics').then(r=>r.json()).then(d => {
  document.getElementById('kpi-spend').textContent = '$' + (d.total_spend/1e6).toFixed(1) + 'M';
  document.getElementById('kpi-invoices').textContent = d.total_invoices || '-';
  document.getElementById('kpi-vendors').textContent = Object.keys(d.vendor_spend||{}).length;
});
fetch('/api/anomalies').then(r=>r.json()).then(d => {
  document.getElementById('kpi-anomalies').textContent = d.summary?.total || 0;
  const list = document.getElementById('anomaly-list');
  const items = (d.anomalies||[]).slice(0,8);
  list.innerHTML = items.map(a => `<div class="anomaly-item"><span class="sev ${a.severity}">${a.severity.toUpperCase()}</span><span>${a.type}: ${a.detail.slice(0,60)}</span></div>`).join('') || 'No anomalies';
});

// ── Chat ───────────────────────────────────────────────────────────
function sendQuery() {
  const q = inputEl.value.trim();
  if (!q || isStreaming) return;
  inputEl.value = '';
  addMessage('user', q);
  streamAgent(q);
  // Also update graph
  updateGraph(q);
}
inputEl.addEventListener('keydown', e => { if (e.key === 'Enter') sendQuery(); });

function addMessage(role, content) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.innerHTML = role === 'user' ? content : `<div class="answer">${content}</div>`;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

function streamAgent(query) {
  isStreaming = true;
  sendBtn.disabled = true;

  const msgDiv = document.createElement('div');
  msgDiv.className = 'msg agent';
  const reasoningDiv = document.createElement('div');
  reasoningDiv.className = 'reasoning';
  const answerDiv = document.createElement('div');
  answerDiv.className = 'answer';
  msgDiv.appendChild(reasoningDiv);
  msgDiv.appendChild(answerDiv);
  messagesEl.appendChild(msgDiv);

  const es = new EventSource('/api/ask?q=' + encodeURIComponent(query));
  es.onmessage = function(e) {
    try {
      const data = JSON.parse(e.data);
      if (data.step === 'done') {
        es.close();
        isStreaming = false;
        sendBtn.disabled = false;
        return;
      }
      if (data.step === 'answer') {
        answerDiv.innerHTML = formatAnswer(data.content);
      } else if (data.step === 'thinking') {
        addStep(reasoningDiv, 'think', data.content);
      } else if (data.step === 'memory') {
        addStep(reasoningDiv, 'brain', data.content);
      } else if (data.step === 'tools_selected') {
        addStep(reasoningDiv, 'tool', 'Tools: ' + (data.tools||[]).join(', '));
      } else if (data.step === 'tool_call') {
        addStep(reasoningDiv, 'search', `Calling ${data.tool}...`);
      } else if (data.step === 'tool_result') {
        addStep(reasoningDiv, 'result', `${data.tool}: ${data.summary}`);
      } else if (data.step === 'memory_stored') {
        addStep(reasoningDiv, 'save', 'Finding stored in knowledge graph');
      }
      messagesEl.scrollTop = messagesEl.scrollHeight;
    } catch(err) {}
  };
  es.onerror = function() {
    es.close();
    isStreaming = false;
    sendBtn.disabled = false;
    if (!answerDiv.innerHTML) answerDiv.innerHTML = '<em style="color:#ef4444">Connection error. Please try again.</em>';
  };
}

const ICONS = {think:'&#9881;', brain:'&#129504;', tool:'&#128295;', search:'&#128269;', result:'&#9989;', save:'&#128190;'};
function addStep(container, icon, text) {
  const div = document.createElement('div');
  div.className = 'step';
  div.innerHTML = `<span class="icon">${ICONS[icon]||'&#8226;'}</span><span>${text}</span>`;
  container.appendChild(div);
}

function formatAnswer(text) {
  if (!text) return '';
  return text
    .replace(/\\n/g, '<br>')
    .replace(/\n/g, '<br>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\$([\d,.]+)/g, '<strong style="color:#f59e0b">$$$1</strong>')
    .replace(/(INV-[\w-]+|TX-[\w-]+|tx-[\w-]+)/g, '<code style="color:#a5b4fc">$1</code>');
}

// ── Knowledge Graph ────────────────────────────────────────────────
function updateGraph(query) {
  fetch('/api/graph?q=' + encodeURIComponent(query))
    .then(r => r.json())
    .then(data => renderGraph(data))
    .catch(() => {});
}

function renderGraph(data) {
  const container = document.getElementById('graph-container');
  const placeholder = document.getElementById('graph-placeholder');
  if (placeholder) placeholder.style.display = 'none';

  const svg = d3.select('#graph-container svg');
  svg.selectAll('*').remove();
  const width = container.clientWidth;
  const height = container.clientHeight;

  if (!data.nodes || data.nodes.length === 0) return;

  const simulation = d3.forceSimulation(data.nodes)
    .force('link', d3.forceLink(data.links).id(d => d.id).distance(60).strength(0.3))
    .force('charge', d3.forceManyBody().strength(-120))
    .force('center', d3.forceCenter(width/2, height/2))
    .force('collide', d3.forceCollide().radius(20));

  const g = svg.append('g');

  // Zoom
  svg.call(d3.zoom().scaleExtent([0.3, 4]).on('zoom', () => g.attr('transform', d3.event.transform)));

  // Links
  const link = g.append('g').selectAll('line').data(data.links).enter().append('line')
    .attr('stroke', '#444').attr('stroke-width', 1.5).attr('stroke-opacity', 0.6);

  // Edge labels
  const edgeLabel = g.append('g').selectAll('text').data(data.links).enter().append('text')
    .text(d => d.relation).attr('font-size', '6px').attr('fill', '#555').attr('text-anchor', 'middle');

  // Nodes
  const node = g.append('g').selectAll('circle').data(data.nodes).enter().append('circle')
    .attr('r', d => d.type === 'Vendor' ? 12 : d.type === 'Invoice' || d.type === 'Transaction' ? 10 : 7)
    .attr('fill', d => d.color || '#7c3aed')
    .attr('stroke', '#fff').attr('stroke-width', 0.5)
    .call(d3.drag()
      .on('start', d => { if (!d3.event.active) simulation.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; })
      .on('drag', d => { d.fx=d3.event.x; d.fy=d3.event.y; })
      .on('end', d => { if (!d3.event.active) simulation.alphaTarget(0); d.fx=null; d.fy=null; }));

  // Node labels
  const label = g.append('g').selectAll('text').data(data.nodes).enter().append('text')
    .text(d => d.name?.length > 20 ? d.name.slice(0,18) + '..' : d.name)
    .attr('font-size', '7px').attr('fill', '#ccc').attr('dx', 14).attr('dy', 3);

  simulation.on('tick', () => {
    link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y).attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
    edgeLabel.attr('x',d=>(d.source.x+d.target.x)/2).attr('y',d=>(d.source.y+d.target.y)/2);
    node.attr('cx',d=>d.x).attr('cy',d=>d.y);
    label.attr('x',d=>d.x).attr('y',d=>d.y);
  });

  // Animate nodes appearing
  node.attr('opacity', 0).transition().duration(600).delay((d,i) => i * 40).attr('opacity', 1);
  label.attr('opacity', 0).transition().duration(600).delay((d,i) => i * 40).attr('opacity', 1);
  link.attr('opacity', 0).transition().duration(400).delay(300).attr('opacity', 0.6);
  edgeLabel.attr('opacity', 0).transition().duration(400).delay(300).attr('opacity', 1);
}

// ── Red Team ───────────────────────────────────────────────────────
async function startRedTeam() {
  addMessage('user', '[Red Team Mode] Generate & detect fraudulent invoice');
  const msgDiv = document.createElement('div');
  msgDiv.className = 'msg agent';
  const reasoningDiv = document.createElement('div');
  reasoningDiv.className = 'reasoning';
  const answerDiv = document.createElement('div');
  answerDiv.className = 'answer';
  msgDiv.appendChild(reasoningDiv);
  msgDiv.appendChild(answerDiv);
  messagesEl.appendChild(msgDiv);

  addStep(reasoningDiv, 'think', 'Generating synthetic fraudulent invoice via Distil Labs SLM...');
  messagesEl.scrollTop = messagesEl.scrollHeight;

  try {
    const genRes = await fetch('/api/redteam/generate', {method:'POST'});
    const genData = await genRes.json();
    addStep(reasoningDiv, 'result', 'Invoice generated');
    addStep(reasoningDiv, 'search', 'Ingesting through cognee pipeline & running detection...');
    messagesEl.scrollTop = messagesEl.scrollHeight;

    const detectRes = await fetch('/api/redteam/detect', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({invoice: genData.invoice})});
    const detectData = await detectRes.json();

    const icon = detectData.detected ? '&#128680;' : '&#9888;';
    const status = detectData.detected ? '<strong style="color:#22c55e">DETECTED</strong>' : '<strong style="color:#ef4444">MISSED</strong>';
    answerDiv.innerHTML = `
      <div style="margin-bottom:0.5rem">${icon} Red Team Result: ${status}</div>
      <div style="background:#111;border-radius:8px;padding:0.75rem;margin-bottom:0.75rem;font-family:monospace;font-size:0.8rem;color:#ccc;white-space:pre-wrap">${genData.invoice}</div>
      <div style="color:#888;font-size:0.8rem;margin-bottom:0.5rem"><em>Hidden fraud: ${genData.fraud_hint || 'unknown'}</em></div>
      <div>${formatAnswer(detectData.explanation)}</div>`;
  } catch(err) {
    answerDiv.innerHTML = '<em style="color:#ef4444">Red Team error: ' + err.message + '</em>';
  }
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ── Ingest Modal ───────────────────────────────────────────────────
function showIngestModal() { document.getElementById('ingestModal').classList.add('show'); }
function closeIngestModal() { document.getElementById('ingestModal').classList.remove('show'); }
async function submitIngest() {
  const text = document.getElementById('ingestText').value.trim();
  if (!text) return;
  closeIngestModal();
  addMessage('user', '[Ingest] Adding new document to knowledge graph...');

  const msgDiv = addMessage('agent', '<em>Processing through cognee pipeline...</em>');
  try {
    const res = await fetch('/api/ingest', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text})});
    const data = await res.json();
    msgDiv.querySelector('.answer').innerHTML = `Document ingested: <strong>${data.status}</strong> (${data.time_ms}ms, ${data.chunks_found} chunks found)`;
  } catch(err) {
    msgDiv.querySelector('.answer').innerHTML = '<em style="color:#ef4444">Ingest error: ' + err.message + '</em>';
  }
  document.getElementById('ingestText').value = '';
}

// ── Memory count ───────────────────────────────────────────────────
fetch('/api/history').then(r=>r.json()).then(d => {
  document.getElementById('memory-count').textContent = (d.history||[]).length + ' conversation entries';
}).catch(() => { document.getElementById('memory-count').textContent = 'Ready'; });
</script>
</body></html>"""


# ── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
