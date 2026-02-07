# ProcureGuard — AI Procurement Auditor with Structured Memory

## Hackathon Strategy & Implementation Plan

---

## Table of Contents

1. [Hackathon Requirements](#hackathon-requirements)
2. [The Four Pillars](#the-four-pillars)
3. [The Winning Idea](#the-winning-idea)
4. [Architecture](#architecture)
5. [Cognee: The Memory Engine](#cognee-the-memory-engine)
6. [Qdrant: The Vector Knowledge Store](#qdrant-the-vector-knowledge-store)
7. [Distil Labs: The Reasoning Brain](#distil-labs-the-reasoning-brain)
8. [DigitalOcean: Production Deployment](#digitalocean-production-deployment)
9. [Feature Breakdown](#feature-breakdown)
10. [Technical Implementation](#technical-implementation)
11. [Implementation Timeline](#implementation-timeline)
12. [Demo Script](#demo-script)
13. [Prize Strategy](#prize-strategy)

---

## Hackathon Requirements

The hackathon mandates a specific data flow. Every project MUST demonstrate this pipeline:

```
Raw Data (invoices, transactions, vendors)
    │
    ▼
COGNEE — Entity extraction, relationship mapping, structured memory
    │
    ▼
QDRANT — Store graph-vector-representations as searchable vectors
    │
    ▼
DISTIL LABS — SLM reasons over the structured knowledge
    │
    ▼
DIGITALOCEAN — Host and share the application
```

This is not optional. Each sponsor must be used **effectively** — not just mentioned, but central to the architecture.

### What the Judges Are Evaluating

| Requirement | What it means concretely |
|---|---|
| **Financial application with memory** | Q&A system that remembers past interactions and builds knowledge over time |
| **Q&A over vendor, product, payment, order info** | Natural language queries answered with specific IDs, amounts, dates |
| **Cognee identifies entities** | `cognee.cognify()` extracts Invoice, Vendor, Product, LineItem, etc. |
| **Cognee extracts relationships** | `issued_by`, `paid_to`, `contains_item`, `refers_to`, `matches_to` |
| **Cognee turns raw data into structured memory** | Raw text → knowledge graph with typed entities and edges |
| **Qdrant stores graph-vector-representations** | All 6 collections: DocumentChunks, Entities, EntityTypes, EdgeTypes, Documents, Summaries |
| **Distil Labs reasons with SLMs** | Their Qwen3-4B fine-tuned model does all inference — tool selection, synthesis, analysis |
| **DigitalOcean hosts the application** | Deployed to DO App Platform, shareable URL |

---

## The Four Pillars

### Pillar 1: Cognee — "The Memory"

Cognee is NOT just a RAG tool. It's the **structured memory engine** that transforms raw procurement documents into a queryable knowledge graph.

**What cognee does in our pipeline:**

```
Raw text: "Invoice INV-V3-M03-200261 from Vendor 3, total $13,699.77
           7x Lenovo ThinkPad X1 Carbon (SKU: PTD-LAP-003) at $2,199 each"
           
                    │ cognee.add() + cognee.cognify()
                    ▼

Entities extracted:
  - Invoice "INV-V3-M03-200261" (type: Invoice)
  - "Vendor 3" (type: Vendor)
  - "Total 13699.77" (type: TotalAmount)
  - "LineItem INV-V3-M03-200261_PTD-LAP-003" (type: LineItem)
  - "Product PTD-LAP-003" (type: Product, name: "Lenovo ThinkPad X1 Carbon")
  - "Quantity 7" (type: Quantity)
  - "Date 2025-03-14" (type: Date)

Relationships extracted:
  - INV-V3-M03-200261 --issued_by--> Vendor 3
  - INV-V3-M03-200261 --has_total--> Total 13699.77
  - INV-V3-M03-200261 --issued_on--> Date 2025-03-14
  - INV-V3-M03-200261 --contains_item--> LineItem INV-V3-M03-200261_PTD-LAP-003
  - LineItem --refers_to--> Product PTD-LAP-003
  - LineItem --has_quantity--> Quantity 7
```

**Why it matters for winning:**
- cognee's CEO is a judge — he wants to see `cognify()` used meaningfully, not just `add()`
- The knowledge graph structure enables relationship-aware queries that pure vector search cannot
- Entity deduplication via shared nodes (same Vendor, Product, Quantity across documents)
- Invoice-Transaction matching via `matches_to` edges

### Pillar 2: Qdrant — "The Store"

Qdrant stores all the graph-vector-representations that cognee produces. But we go beyond basic storage — we use **7 advanced Qdrant APIs** to query the knowledge.

**Collections (created by cognee → stored in Qdrant):**

| Collection | Records | What it stores | How we query it |
|---|---|---|---|
| `DocumentChunk_text` | 2,000 | Invoice/transaction text chunks | Prefetch + RRF Fusion, Discovery, Recommend |
| `Entity_name` | 8,816 | Products, vendors, SKUs, amounts | Filtered search, Group API |
| `EntityType_name` | 8 | Entity type definitions | Type lookups |
| `EdgeType_relationship_name` | 13 | Relationship types (issued_by, paid_to...) | Relationship queries |
| `TextDocument_name` | 2,000 | Document references | Document retrieval |
| `TextSummary_text` | 2,000 | Document summaries | Summary search |

**Advanced APIs we use:**

| Qdrant Feature | Use Case in ProcureGuard |
|---|---|
| **Prefetch + RRF Fusion** | Every search runs multi-stage retrieval with two candidate pools |
| **Discovery API** | "Find invoices like this one but NOT from Vendor 12" — context-aware search |
| **Recommend API** | "Show records similar to this anomaly" — anomaly investigation |
| **Batch Query API** | Scan 200 records for near-duplicates in batches of 50 |
| **Group API** | Faceted results by vendor, product type, date |
| **Scroll API** | Bulk data extraction for analytics aggregation |
| **Payload Indexing** | Fast filtering on vendor_id, type, date fields |

### Pillar 3: Distil Labs — "The Brain"

The Distil Labs fine-tuned Qwen3-4B SLM is the reasoning engine for ALL intelligence in the app. No cloud LLM calls — everything runs locally through their model.

**Model specs:**
- `cognee-distillabs-model-gguf-quantized` (2.5 GB, Qwen3 4B, Q4_K_M quantization)
- Running via Ollama on `localhost:11434`
- OpenAI-compatible API (`/v1/chat/completions`)

**What the SLM does:**
1. **Tool selection** — Decides which tools to call based on user query
2. **Answer synthesis** — Combines results from multiple tool calls into coherent responses
3. **Pattern inference** — Detects discrepancies, trends, anomalies from data patterns
4. **Risk assessment** — Evaluates severity of findings
5. **Memory-aware reasoning** — Uses cognee context to cross-reference across sessions
6. **Report generation** — Structured audit findings with specific numbers

**Critical**: The system prompt from `prompts/system_prompt.txt` shows exactly what the judges expect:
- **Verification**: "Check whether all payments to Vendor 2 are correct" → must cite specific IDs and amounts
- **Lookup**: "Did we buy any storage devices?" → must list transaction IDs, dates, SKUs, quantities
- **Comparison**: "Do we spend more with Vendor 4 or Vendor 2?" → must provide exact dollar comparison
- **Analysis**: "Which vendors consistently give discounts?" → must INFER patterns from multiple data points

### Pillar 4: DigitalOcean — "The Platform"

The app must be deployable and shareable. DigitalOcean App Platform provides production hosting.

**Deployment architecture:**
- App Platform container with `LLM_MODE=remote` and `EMBED_MODE=remote`
- Qdrant Cloud or DO-hosted Qdrant for vector storage
- DO Spaces for model files and snapshot storage
- Single unified app (not 3 separate services)

---

## The Winning Idea

### ProcureGuard: An AI Procurement Auditor with Structured Memory

A conversational financial agent that:

1. **Ingests raw procurement data** through cognee's entity extraction pipeline
2. **Stores structured knowledge** as graph-vector-representations in Qdrant
3. **Reasons over the knowledge** using Distil Labs' SLM
4. **Remembers investigations** — findings and conversations persist in the knowledge graph
5. **Answers financial questions** with specific IDs, amounts, dates, and inferred patterns
6. **Runs in production** on DigitalOcean App Platform

### Why This Wins

It's not a search UI. It's not a dashboard. It's an **intelligent agent with memory** that demonstrates the full cognee → Qdrant → Distil Labs pipeline in a way that solves a real financial problem.

---

## Architecture

### Complete Data Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                        RAW DATA SOURCES                          │
│  Invoices (1,000)  │  Transactions (1,000)  │  New documents     │
│  CSV / Text files  │  CSV / Text files      │  User-added text   │
└────────┬───────────┴──────────┬─────────────┴──────────┬─────────┘
         │                      │                        │
         ▼                      ▼                        ▼
┌──────────────────────────────────────────────────────────────────┐
│                     COGNEE — Memory Engine                        │
│                                                                    │
│  cognee.add(raw_text)                                             │
│       │                                                            │
│       ▼                                                            │
│  cognee.cognify()  ──→  Entity Extraction (LLM-powered)           │
│       │                  - Invoice, Transaction, Vendor            │
│       │                  - Product (by SKU), LineItem, Quantity    │
│       │                  - TotalAmount, Date                       │
│       │                                                            │
│       ├──→  Relationship Extraction                                │
│       │     - issued_by, paid_to, contains_item                   │
│       │     - refers_to, has_quantity, has_total                   │
│       │     - matches_to (invoice ↔ transaction matching)         │
│       │                                                            │
│       ├──→  Summary Generation                                     │
│       │     - Document summaries for fast retrieval                │
│       │                                                            │
│       └──→  Vector Embedding (nomic-embed-text, 768-dim)          │
│                                                                    │
│  cognee.search(query, SearchType)                                 │
│       - CHUNKS: Raw document chunks                                │
│       - SUMMARIES: Document summaries                              │
│       - GRAPH_COMPLETION: LLM reasoning with graph context         │
│       - RAG_COMPLETION: Traditional RAG                            │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                   QDRANT — Vector Knowledge Store                  │
│                                                                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐ │
│  │ DocumentChunk   │  │ Entity_name     │  │ TextSummary      │ │
│  │ _text (2,000)   │  │ (8,816)         │  │ _text (2,000)    │ │
│  │                 │  │                 │  │                  │ │
│  │ Invoice chunks  │  │ Vendors,        │  │ Document         │ │
│  │ Transaction     │  │ Products (SKU), │  │ summaries        │ │
│  │ chunks          │  │ LineItems,      │  │                  │ │
│  │                 │  │ Amounts, Dates  │  │                  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────────┘ │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐ │
│  │ TextDocument    │  │ EntityType      │  │ EdgeType         │ │
│  │ _name (2,000)   │  │ _name (8)       │  │ _relationship    │ │
│  │                 │  │                 │  │ _name (13)       │ │
│  │ Doc references  │  │ Type defs       │  │ Relationship     │ │
│  │                 │  │                 │  │ types            │ │
│  └─────────────────┘  └─────────────────┘  └──────────────────┘ │
│                                                                    │
│  Query APIs: Prefetch+Fusion, Discovery, Recommend, Batch,        │
│              Group, Scroll, Payload Filtering                      │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                  DISTIL LABS — Reasoning Brain                      │
│                                                                    │
│  cognee-distillabs-model-gguf-quantized (Qwen3 4B, Q4_K_M)       │
│  Running via Ollama on localhost:11434                              │
│                                                                    │
│  Capabilities:                                                     │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│  │ Tool         │ │ Answer       │ │ Pattern                  │ │
│  │ Selection    │ │ Synthesis    │ │ Inference                │ │
│  │              │ │              │ │                          │ │
│  │ "Which tools │ │ "Combine 4   │ │ "Vendor 1 has varying   │ │
│  │ do I need    │ │ tool results │ │ amounts across 2 txns   │ │
│  │ for this     │ │ into clear   │ │ → payment discrepancy"  │ │
│  │ question?"   │ │ answer"      │ │                          │ │
│  └──────────────┘ └──────────────┘ └──────────────────────────┘ │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│  │ Risk         │ │ Memory-Aware │ │ Audit Report             │ │
│  │ Assessment   │ │ Reasoning    │ │ Generation               │ │
│  │              │ │              │ │                          │ │
│  │ "High sev:   │ │ "Last time   │ │ "Finding #1: 3 duplicate │ │
│  │ $89K outlier │ │ we found     │ │ invoices totaling $47K  │ │
│  │ in INV-0042" │ │ Vendor 12..."│ │ from Vendor 12..."      │ │
│  └──────────────┘ └──────────────┘ └──────────────────────────┘ │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                DIGITALOCEAN — Production Platform                   │
│                                                                    │
│  App Platform:     Single container, unified ProcureGuard app     │
│  DO Spaces:        Model files, Qdrant snapshots, static assets   │
│  Environment:      LLM_MODE=remote, EMBED_MODE=remote             │
│  Shareable URL:    https://procureguard-xxxxx.ondigitalocean.app  │
└──────────────────────────────────────────────────────────────────┘
```

### Agent Loop Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface                             │
│  ┌───────────────────────┐  ┌──────────────────────────────┐│
│  │     Chat Panel         │  │      Dashboard Sidebar       ││
│  │                        │  │                              ││
│  │  User: "Check if all  │  │  KPIs: $12.4M spend         ││
│  │  payments to Vendor 2 │  │  1,000 invoices              ││
│  │  are correct"          │  │  47 anomalies               ││
│  │                        │  │                              ││
│  │  Agent: [calling       │  │  Top Alerts:                ││
│  │  tools...]             │  │  🔴 INV-0042 amount outlier ││
│  │                        │  │  🔴 3 near-duplicates       ││
│  │  Agent: "Found one     │  │  🟡 Vendor 12 high variance ││
│  │  payment: tx-v2-m02-   │  │                              ││
│  │  176206 for $3,723.76  │  │  Memory Panel:              ││
│  │  with $460.24 discount.│  │  🧠 3 stored findings       ││
│  │  No other payments..." │  │  🧠 Last: Vendor 12 risk   ││
│  └───────────────────────┘  └──────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     Agent Loop                               │
│                                                               │
│  1. User query arrives                                        │
│  2. Recall memory (cognee.search → relevant past findings)   │
│  3. Send to Distil Labs SLM with tool descriptions            │
│  4. SLM selects tool(s) and parameters                        │
│  5. Execute tool(s) against Qdrant                            │
│  6. Return results to SLM                                     │
│  7. SLM synthesizes final answer                              │
│  8. Store finding in cognee (cognee.add → cognee.cognify)    │
│  9. Return answer to user                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Cognee: The Memory Engine

### How We Use Cognee (Beyond the Starter Code)

The starter projects use Qdrant directly. ProcureGuard uses cognee as the **primary interface** for data ingestion and retrieval.

#### 1. Initial Data Ingestion (startup)

The existing 2,000 records are already in Qdrant (processed through cognee during the pre-hackathon setup). This gives us:
- 8,816 entities (vendors, products, invoices, transactions, line items, etc.)
- 13 relationship types connecting them
- 2,000 document summaries

#### 2. Live Data Enrichment (user interaction)

Users can add new procurement data that flows through the full cognee pipeline:

```python
# User adds a new invoice via the UI
raw_text = "Invoice INV-V9-M03-779589 from Vendor 9, total $25,325.84..."

# Step 1: Add to cognee
await cognee.add(raw_text)

# Step 2: Extract entities and relationships
await cognee.cognify()
# This extracts: Invoice entity, Vendor 9, Product SKU SVI-LAP-009,
# LineItems, Quantities, TotalAmount, Dates
# AND creates relationships: issued_by, contains_item, refers_to, etc.
# AND stores everything as vectors in Qdrant

# Step 3: Now searchable via graph-aware retrieval
results = await cognee.search(
    query_text="What did we buy from Vendor 9?",
    query_type=SearchType.GRAPH_COMPLETION
)
# Returns: Microsoft Surface Laptop 4 (16 units), Samsung T7 SSD (8 units)
# WITH relationship context from the knowledge graph
```

#### 3. Investigation Memory (agent behavior)

When the agent investigates an anomaly, it stores the finding as structured memory:

```python
# Agent finds suspicious pattern
finding = """
AUDIT FINDING: Vendor 12 has 3 invoices with identical line items but different totals.
INV-V12-M02-450045 ($1,582.70) and INV-V12-M02-450046 ($1,612.70) both contain
7x Kingston A2000 1TB NVMe (BW-SSD-012) — $30 discrepancy suggests potential billing error.
Severity: HIGH. Recommended action: Request vendor reconciliation.
"""

await cognee.add(finding)
await cognee.cognify()
# Now cognee has extracted: Vendor 12, INV-V12-M02-450045, INV-V12-M02-450046,
# Product BW-SSD-012, the discrepancy relationship, and severity assessment
```

#### 4. Cross-Session Recall (persistent memory)

```python
# Later, user asks about Vendor 12 again
results = await cognee.search(
    query_text="What do we know about Vendor 12?",
    query_type=SearchType.GRAPH_COMPLETION
)
# Returns BOTH: original invoice data AND the stored audit finding
# The SLM can cross-reference: "Previously I found a $30 billing discrepancy..."
```

#### 5. Search Types We Use

| Search Type | When to Use | Example |
|---|---|---|
| `CHUNKS` | Fast retrieval of raw document text | "Show me Invoice INV-V3-M03-200261" |
| `SUMMARIES` | Quick overview of documents | "Summarize all Vendor 9 activity" |
| `GRAPH_COMPLETION` | Relationship-aware reasoning | "Which vendors have payment discrepancies?" |
| `RAG_COMPLETION` | Traditional RAG with context | "How much did we spend on laptops?" |

---

## Qdrant: The Vector Knowledge Store

### How Cognee Uses Qdrant

Cognee stores its knowledge graph as vectors in Qdrant. Each entity, relationship, document chunk, and summary becomes a vector with structured payload.

### How We Query Qdrant (7 Advanced APIs)

#### 1. Prefetch + RRF Fusion (Multi-Stage Retrieval)

Every search runs a two-stage pipeline for better ranking:

```python
results = qdrant.query_points(
    collection_name="DocumentChunk_text",
    prefetch=[
        Prefetch(query=vector, using="text", limit=100),  # broad recall
        Prefetch(query=vector, using="text", limit=50),   # tight precision
    ],
    query=FusionQuery(fusion=Fusion.RRF),  # fuse rankings
    limit=20,
    with_payload=True,
)
```

**Why it matters**: RRF fusion combines broad recall with precise ranking, significantly improving result quality over single-stage search.

#### 2. Discovery API (Context-Aware Investigation)

"Find invoices similar to this flagged one, but NOT from this vendor":

```python
results = qdrant.query_points(
    collection_name="DocumentChunk_text",
    query=DiscoverQuery(
        discover=DiscoverInput(
            target=query_vector,
            context=[ContextPair(positive=good_invoice_id, negative=bad_vendor_id)],
        )
    ),
    using="text",
    limit=10,
)
```

**Why it matters**: Discovery API enables nuanced investigation — steering toward positive examples while avoiding negative ones. Essential for cross-vendor fraud pattern detection.

#### 3. Recommend API (Anomaly Investigation)

"Find records similar to this anomaly":

```python
results = qdrant.query_points(
    collection_name="DocumentChunk_text",
    query=RecommendQuery(
        recommend=RecommendInput(
            positive=[anomaly_point_id],
            strategy=RecommendStrategy.BEST_SCORE,
        )
    ),
    using="text",
    limit=10,
)
```

**Why it matters**: When an anomaly is flagged, Recommend API finds similar patterns across the entire dataset — critical for determining if it's a one-off error or systemic issue.

#### 4. Batch Query API (Duplicate Detection)

Scan hundreds of records for near-duplicates efficiently:

```python
from qdrant_client.models import QueryRequest

requests = [
    QueryRequest(query=r["id"], using="text", limit=3, score_threshold=0.99)
    for r in batch  # 50 records per batch
]
responses = qdrant.query_batch_points(
    collection_name="DocumentChunk_text",
    requests=requests,
    timeout=30,
)
```

**Why it matters**: Duplicate invoices are a top procurement fraud vector. Batch queries make scanning 200+ records feasible in seconds.

#### 5. Group API (Faceted Analysis)

Results grouped by vendor, product type, or date:

```python
groups = qdrant.query_points_groups(
    collection_name="DocumentChunk_text",
    query=vector,
    using="text",
    group_by="type",
    limit=20,
    group_size=5,
    with_payload=True,
)
```

**Why it matters**: Shows spend breakdown by vendor or product category in a single query.

#### 6. Scroll API (Analytics Data Loading)

Bulk-load all records for statistical analysis:

```python
points, offset = qdrant.scroll(
    collection_name="DocumentChunk_text",
    limit=250,
    offset=offset,
    with_payload=True,
    with_vectors=True,  # needed for vector outlier detection
)
```

**Why it matters**: Powers the analytics dashboard and anomaly detection (z-score calculation requires all data points).

#### 7. Payload-Indexed Filtering

Fast filtered queries on indexed fields:

```python
results = qdrant.query_points(
    collection_name="DocumentChunk_text",
    query=vector,
    using="text",
    query_filter=Filter(
        must=[FieldCondition(key="type", match=MatchValue(value="invoice"))]
    ),
    limit=20,
)
```

**Why it matters**: Enables instant filtering by vendor, type, date without full collection scan.

---

## Distil Labs: The Reasoning Brain

### The SLM Powers ALL Intelligence

Every decision in ProcureGuard flows through the Distil Labs model. No cloud LLM calls.

### Agent Tool Selection

The SLM receives the user query plus tool descriptions and decides what to call:

```python
TOOLS = {
    "search_knowledge": {
        "description": "Search the knowledge graph for invoices, transactions, vendors, products. "
                       "Uses cognee graph-aware search + Qdrant Prefetch/Fusion.",
        "params": {"query": "str", "search_type": "CHUNKS|SUMMARIES|GRAPH_COMPLETION", "collection": "str"},
    },
    "analyze_spend": {
        "description": "Get spend analytics: vendor totals, monthly trends, product breakdown, comparisons.",
        "params": {"vendor_id": "int (optional)", "compare_vendors": "list (optional)"},
    },
    "detect_anomalies": {
        "description": "Check for anomalies: amount outliers, duplicates, vector outliers, vendor variance.",
        "params": {"vendor_id": "int (optional)", "anomaly_type": "str (optional)"},
    },
    "investigate_record": {
        "description": "Deep investigation of a record: find similar records via Qdrant Recommend/Discovery API.",
        "params": {"record_id": "str", "exclude_vendor": "int (optional)"},
    },
    "recall_memory": {
        "description": "Search past investigations and findings stored in the knowledge graph via cognee.",
        "params": {"query": "str"},
    },
    "store_finding": {
        "description": "Store an audit finding or note in the knowledge graph via cognee.add() + cognify().",
        "params": {"finding_text": "str"},
    },
    "add_document": {
        "description": "Ingest a new procurement document through cognee's entity extraction pipeline.",
        "params": {"raw_text": "str"},
    },
}
```

### Question Types the SLM Handles

Based on the hackathon's `system_prompt.txt`, the SLM must handle:

| Question Type | Example | SLM Behavior |
|---|---|---|
| **Verification** | "Check whether all payments to Vendor 2 are correct" | Search for Vendor 2 txns → list specific IDs and amounts → note missing data |
| **Lookup** | "Did we buy any storage devices?" | Search for HDDs, SSDs → list each with TX-ID, date, SKU, qty, amount |
| **Comparison** | "Do we spend more with Vendor 4 or Vendor 2?" | Pull both vendors' totals → compute and compare → "$17,161 vs $3,724" |
| **Analysis** | "Which vendors consistently give discounts?" | Search all txns → find vendors with 2+ discount transactions → infer pattern |
| **Investigation** | "Investigate suspicious patterns with Vendor 12" | Chain: search → anomaly check → similar records → risk assessment → store finding |
| **Memory recall** | "What did you find about Vendor 7 earlier?" | Query cognee for past findings → cross-reference with current context |

### Inference Requirement (Critical for Scoring)

The judges explicitly test **pattern inference**. The SLM must NOT say "the graph contains no information about X" when X can be inferred.

**Example — Payment discrepancies:**
- BAD: "The graph does not contain information about payment discrepancies." (score: 0)
- GOOD: "Vendor 1 has two transactions with differing amounts ($1,394.88 and $1,364.88) — this $30 variance suggests a potential payment discrepancy." (score: 1)

The system prompt with inference rules should be baked into every SLM call.

---

## DigitalOcean: Production Deployment

### Deployment Architecture

```
┌─────────────────────────────────────────────┐
│          DigitalOcean App Platform            │
│                                               │
│  ┌─────────────────────────────────────────┐ │
│  │  ProcureGuard Container                  │ │
│  │  - FastAPI app (single unified service) │ │
│  │  - LLM_MODE=remote                       │ │
│  │  - EMBED_MODE=remote                     │ │
│  │  - Points to hosted Ollama or LLM API   │ │
│  └─────────────────────────────────────────┘ │
│                                               │
│  Environment variables (secrets):             │
│  - QDRANT_URL, QDRANT_API_KEY               │
│  - LLM_API_URL, LLM_API_KEY                │
│  - EMBED_API_URL, EMBED_API_KEY             │
│                                               │
│  Routes: / → ProcureGuard UI                 │
└──────────────┬──────────────────────────────┘
               │
     ┌─────────┼─────────┐
     ▼         ▼         ▼
┌─────────┐ ┌──────┐ ┌────────┐
│ Qdrant  │ │ LLM  │ │  DO    │
│ (local  │ │ API  │ │Spaces  │
│ or      │ │      │ │        │
│ Cloud)  │ │      │ │Models, │
│         │ │      │ │Assets  │
└─────────┘ └──────┘ └────────┘
```

### DO App Platform Spec (Updated for Unified App)

```yaml
name: procureguard
services:
  - name: procureguard
    github:
      repo: <your-repo>
      branch: main
      deploy_on_push: true
    dockerfile_path: Dockerfile
    envs:
      - key: LLM_MODE
        value: remote
      - key: EMBED_MODE
        value: remote
      - key: QDRANT_URL
        type: SECRET
      - key: QDRANT_API_KEY
        type: SECRET
      - key: LLM_API_URL
        type: SECRET
      - key: LLM_API_KEY
        type: SECRET
      - key: LLM_MODEL_NAME
        value: cognee-distillabs-model-gguf-quantized
      - key: EMBED_API_URL
        type: SECRET
      - key: EMBED_MODEL_NAME
        value: nomic-embed-text
    http_port: 8000
    instance_count: 1
    instance_size_slug: apps-s-1vcpu-2gb
    routes:
      - path: /
```

### DO Spaces Usage

```python
# Upload models and snapshots to DO Spaces
# (already scripted in upload-to-spaces.py)
import boto3

s3 = boto3.client('s3',
    endpoint_url=os.environ['SPACES_ENDPOINT'],
    aws_access_key_id=os.environ['SPACES_KEY'],
    aws_secret_access_key=os.environ['SPACES_SECRET'],
)
s3.upload_file('snapshots/DocumentChunk_text.snapshot', BUCKET, 'snapshots/DocumentChunk_text.snapshot')
```

---

## Feature Breakdown

### Must-Have Features (Core — 4 hours)

| # | Feature | Sponsor Highlighted | Implementation |
|---|---|---|---|
| 1 | **Cognee data ingestion pipeline** | Cognee | `cognee.add()` + `cognee.cognify()` for new documents |
| 2 | **Graph-aware Q&A** | Cognee + Distil Labs | `cognee.search(GRAPH_COMPLETION)` → SLM synthesis |
| 3 | **Multi-tool agent loop** | Distil Labs | SLM selects and chains tools based on query |
| 4 | **Advanced Qdrant search** | Qdrant | Prefetch+Fusion, Discovery, Recommend, Group, Filter |
| 5 | **Anomaly detection** | Qdrant | Amount outliers, vector outliers, duplicates, vendor variance |
| 6 | **Persistent memory** | Cognee | Store/recall findings via `cognee.add()` + `cognee.search()` |
| 7 | **Chat UI with dashboard** | All | Beautiful interface showing all capabilities |

### Nice-to-Have Features (Polish — 1.5 hours)

| # | Feature | Sponsor Highlighted | Implementation |
|---|---|---|---|
| 8 | **Live document ingestion** | Cognee | User pastes new invoice → full cognee pipeline in real-time |
| 9 | **Investigation workflow** | Qdrant + Distil Labs | Auto-chain: search → anomaly → recommend → risk assessment |
| 10 | **Audit report export** | Distil Labs | SLM generates structured audit report from findings |
| 11 | **DO deployment** | DigitalOcean | Live on App Platform with shareable URL |
| 12 | **New data enrichment** | Cognee | Ingest `new_invoices.csv` through cognee pipeline |

---

## Technical Implementation

### Project Structure

```
procureguard/
├── app.py              # Unified FastAPI app
├── agent.py            # Agent loop (tool selection, execution, synthesis)
├── tools/
│   ├── search.py       # Qdrant search tools (Fusion, Discovery, Recommend, Group)
│   ├── analytics.py    # Spend analytics (Scroll + aggregation)
│   ├── anomaly.py      # Anomaly detection (outliers, duplicates, vendor variance)
│   └── memory.py       # Cognee integration (add, cognify, search, recall)
├── pyproject.toml
├── .env
└── Dockerfile
```

### Agent Loop Implementation

```python
import json
from shared.llm import get_llm_response
from tools import search, analytics, anomaly, memory

SYSTEM_PROMPT = """You are ProcureGuard, an AI procurement auditor with persistent memory.

You have these tools:
{tool_descriptions}

To use a tool, respond with EXACTLY this format:
TOOL: tool_name
PARAMS: {{"key": "value"}}

You can call multiple tools by including multiple TOOL/PARAMS blocks.
After receiving tool results, provide your final answer.

CRITICAL RULES:
- Always cite specific IDs, amounts, dates, SKUs
- INFER patterns from data — never say "no information about X" if X can be deduced
- For comparisons, provide exact numbers
- For investigations, chain multiple tools
- Store important findings for future reference
"""

async def agent_respond(user_query: str, conversation_history: list) -> dict:
    # Step 1: Recall relevant memory from cognee
    memory_context = await memory.recall(user_query)

    # Step 2: Ask SLM to select tools
    prompt = f"""Memory context: {memory_context}

Conversation history: {format_history(conversation_history)}

User question: {user_query}

Select the tools needed to answer this question."""

    tool_response = get_llm_response(SYSTEM_PROMPT, prompt)

    # Step 3: Parse and execute tools
    tool_calls = parse_tool_calls(tool_response)
    tool_results = {}
    for tool_name, params in tool_calls:
        result = await execute_tool(tool_name, params)
        tool_results[tool_name] = result

    # Step 4: Synthesize final answer
    synthesis_prompt = f"""User question: {user_query}

Tool results:
{json.dumps(tool_results, indent=2, default=str)}

Memory context: {memory_context}

Provide a clear, specific answer. Cite IDs, amounts, dates. Infer patterns when needed."""

    answer = get_llm_response(SYSTEM_PROMPT, synthesis_prompt)

    # Step 5: Store finding if significant
    if is_significant_finding(answer, tool_results):
        await memory.store_finding(f"Q: {user_query}\nA: {answer}")

    return {"answer": answer, "tools_used": list(tool_results.keys()), "memory_context": memory_context}
```

### Cognee Memory Tool

```python
import cognee
from cognee.api.v1.search import SearchType

async def recall(query: str) -> str:
    """Search the knowledge graph for relevant past findings and data."""
    try:
        results = await cognee.search(
            query_text=query,
            query_type=SearchType.GRAPH_COMPLETION,
        )
        return "\n".join(str(r)[:300] for r in results[:5])
    except Exception:
        return ""

async def store_finding(text: str):
    """Store a new finding in the knowledge graph."""
    await cognee.add(text)
    await cognee.cognify()

async def ingest_document(raw_text: str) -> dict:
    """Ingest a new procurement document through the full cognee pipeline."""
    await cognee.add(raw_text)
    await cognee.cognify()
    # Verify ingestion
    results = await cognee.search(
        query_text=raw_text[:100],
        query_type=SearchType.CHUNKS,
    )
    return {"status": "ingested", "chunks_created": len(results)}

async def search_chunks(query: str) -> list:
    return await cognee.search(query_text=query, query_type=SearchType.CHUNKS)

async def search_summaries(query: str) -> list:
    return await cognee.search(query_text=query, query_type=SearchType.SUMMARIES)

async def search_graph(query: str) -> list:
    return await cognee.search(query_text=query, query_type=SearchType.GRAPH_COMPLETION)
```

---

## Implementation Timeline

### Hackathon Day (10:30 AM - 4:00 PM = 5.5 hours)

#### Phase 1: Unified App + Cognee Pipeline (10:30 - 11:30) — 1 hour

**Goal**: Single FastAPI app with cognee as the memory engine.

- [ ] Create `procureguard/app.py` — unified FastAPI app on port 8000
- [ ] Set up cognee integration: configure Qdrant adapter, LLM, embeddings
- [ ] Implement `/ingest` endpoint: `cognee.add()` → `cognee.cognify()` → entities in Qdrant
- [ ] Implement `/search` endpoint: `cognee.search()` with all SearchTypes
- [ ] Test: Add a new invoice → verify entities extracted → search for it
- [ ] Verify cognee → Qdrant pipeline works end-to-end

**Key deliverable**: Raw text in → cognee extracts entities/relationships → stored in Qdrant → searchable

#### Phase 2: Agent + Qdrant Tools (11:30 - 12:30) — 1 hour

**Goal**: Distil Labs SLM selects and executes tools against Qdrant.

- [ ] Implement agent loop: query → SLM tool selection → execute → synthesize
- [ ] Port Qdrant search tools (Prefetch+Fusion, Discovery, Recommend, Group, Filter)
- [ ] Port analytics tools (Scroll + aggregation)
- [ ] Port anomaly detection (amount outliers, vector outliers, duplicates)
- [ ] Run anomaly detection on startup, cache results
- [ ] Test: "Check payments to Vendor 2" → SLM calls search + analytics → specific answer

**Key deliverable**: SLM reasons over Qdrant data with 7 advanced APIs

#### Phase 3: Persistent Memory (12:30 - 1:00) — 30 min

**Goal**: Agent stores and recalls findings via cognee.

- [ ] Implement `store_finding` tool: findings → `cognee.add()` → `cognee.cognify()`
- [ ] Implement `recall_memory` tool: `cognee.search(GRAPH_COMPLETION)`
- [ ] Auto-store significant findings after each investigation
- [ ] Pre-seed 2-3 findings for demo (so memory recall is instant)
- [ ] Test: Investigate Vendor 12 → store finding → ask about Vendor 12 later → recalls

**Key deliverable**: Agent has persistent memory across conversations

#### Lunch (1:00 - 1:30)

Plan demo narrative. Think about edge cases.

#### Phase 4: UI Polish (1:30 - 2:30) — 1 hour

**Goal**: Beautiful, demo-worthy chat interface with dashboard.

- [ ] Chat panel: message history, user/agent styling, loading states
- [ ] Tool execution indicators: "Searching knowledge graph...", "Detecting anomalies..."
- [ ] Dashboard sidebar: KPIs, top anomalies, memory panel
- [ ] "Add Document" feature: paste new invoice → cognee pipeline → confirmation
- [ ] Format results: tables for data, severity badges for anomalies, inline citations
- [ ] Dark theme, responsive, looks good on projector

**Key deliverable**: Demo-worthy UI that showcases all four sponsors

#### Phase 5: DigitalOcean Deployment (2:30 - 3:15) — 45 min

**Goal**: Live on DO with shareable URL.

- [ ] Create Dockerfile for unified ProcureGuard app
- [ ] Update `.do/app.yaml` for single service
- [ ] Set `LLM_MODE=remote`, `EMBED_MODE=remote`
- [ ] Deploy via `doctl apps create --spec .do/app.yaml`
- [ ] Upload assets to DO Spaces
- [ ] Verify live URL works
- [ ] Screenshot DO dashboard for slides

**Key deliverable**: https://procureguard-xxxxx.ondigitalocean.app works

#### Phase 6: Polish + Demo Prep (3:15 - 4:00) — 45 min

**Goal**: Bulletproof demo.

- [ ] Fix bugs found during testing
- [ ] Pre-seed cognee with investigation findings (instant memory demo)
- [ ] Ingest `new_invoices.csv` through cognee pipeline (shows live enrichment)
- [ ] Test the full demo flow 3 times end-to-end
- [ ] Prepare 2-3 slides (architecture, sponsor usage, live demo)
- [ ] Write exact demo queries
- [ ] Submit project

---

## Demo Script

### The 3-Minute Narrative

**Theme**: "From raw data to structured memory to intelligent audit"

#### Act 1: The Pipeline (0:00 - 0:45)

Paste a new invoice into the "Add Document" field:

```
Invoice INV-V9-M03-779589 from Vendor 9, total $25,325.84
10x Microsoft Surface Laptop 4 (SKU: SVI-LAP-009) at $1,699 = $16,990
8x Samsung T7 1TB Portable SSD (SKU: SVI-SSD-009) at $159 = $1,272
```

Watch cognee process it in real-time:
- "Extracting entities..." → Invoice, Vendor 9, Products, LineItems
- "Mapping relationships..." → issued_by, contains_item, refers_to
- "Stored in Qdrant" → vectors with structured payload

**Say**: "ProcureGuard starts with cognee. Raw procurement text goes in, and cognee extracts entities — invoices, vendors, products by SKU, line items with quantities — and maps all the relationships between them. These graph-vector-representations are stored in Qdrant. Let me show you what we can do with this."

#### Act 2: The Investigation (0:45 - 1:45)

Type: **"Check whether all payments to Vendor 2 are correct"**

Watch the agent:
1. "Searching knowledge graph..." (cognee GRAPH_COMPLETION)
2. "Querying Qdrant..." (Prefetch + RRF Fusion)
3. "Checking anomalies..." (amount outlier + duplicate detection)

Agent responds with specific data:
> "Found one payment to Vendor 2: tx-v2-m02-176206 ($3,723.76 with $460.24 discount, recorded 2025-03-04). The invoice INV-V2-M02-828264 shows the same total. However, I notice the discount is 12.4% — significantly above the typical 5% discount rate. Flagging for review."

**Say**: "The Distil Labs SLM chained three tools together — cognee's graph search, Qdrant's multi-stage retrieval with RRF fusion, and our anomaly detector. It didn't just find the payment — it INFERRED that the discount rate is anomalous by comparing patterns across vendors. This is a 4-billion parameter model running entirely locally."

#### Act 3: The Memory (1:45 - 2:30)

Type: **"Which vendors consistently give discounts?"**

Agent recalls the Vendor 2 finding from earlier AND searches all transactions:
> "Based on my analysis and previous investigations: Vendor 1 shows discounts in multiple transactions (TX-V15-M01-588603 and TX-V15-M01-654498). Vendor 2 also gives discounts — I flagged their 12.4% rate as unusually high in my earlier investigation. Vendor 15 shows consistent discounts across 3 invoices."

**Say**: "The agent remembered my earlier finding about Vendor 2 — that's cognee's persistent memory. It stored my investigation as a new entity in the knowledge graph, and now it cross-references past findings with new queries. This is AI with real memory, not just one-shot RAG."

#### Act 4: The Stack (2:30 - 3:00)

Show architecture diagram briefly.

**Say**: "To recap: cognee turns raw procurement text into structured memory — entities, relationships, summaries. Qdrant stores the graph-vector-representations and powers 7 different retrieval strategies. Distil Labs' SLM does all the reasoning — tool selection, pattern inference, answer synthesis. And the whole thing is deployed on DigitalOcean. Four technologies, one intelligent agent."

---

## Prize Strategy

### Why We Win Each Category

| Prize | Our Edge |
|---|---|
| **1st Place ($1,200)** | Only project that's an actual agent with memory — not a search UI or dashboard |
| **Best use of Cognee ($CEO is judge)** | cognee is THE pipeline — entity extraction, relationship mapping, persistent memory, 4 search types |
| **Best use of Qdrant ($1,500)** | 7 advanced APIs (Fusion, Discovery, Recommend, Batch, Group, Scroll, Filter) in one app |
| **Best use of Distil Labs ($1,000)** | SLM does ALL reasoning — tool selection, pattern inference, synthesis, report generation |
| **Best use of DigitalOcean ($1,400+)** | Live on App Platform + DO Spaces for assets, shareable URL |

### What to Say to Each Judge

**To Vasilije (Cognee CEO)**: "We use cognee as the core memory layer — not just for initial ingestion, but for live document enrichment and persistent investigation memory. Every finding gets stored via `cognify()` with full entity extraction, so the agent's knowledge grows over time."

**To Daniel (Cognee Researcher)**: "The agent uses `GRAPH_COMPLETION` search to reason across relationships in the knowledge graph. When it detects a discount anomaly, it follows the `issued_by` and `paid_to` edges to cross-reference invoices and transactions for the same vendor."

**To Thierry (Qdrant DevRel)**: "We use 7 different Qdrant APIs in a single investigation flow. Prefetch+Fusion for every search, Discovery API for context-aware investigation, Recommend for anomaly drill-down, Batch Query for duplicate detection, Group API for vendor faceting."

**To Lizzie (DO DevRel)**: "The app runs on DO App Platform with models on DO Spaces. Single container, remote mode, deployed with one command. The same codebase runs locally for development and in production on DO."

---

## Appendix: Available Data

### Existing Qdrant Collections (from cognee pipeline)

| Collection | Records | Content |
|---|---|---|
| DocumentChunk_text | 2,000 | Invoice + transaction text chunks |
| Entity_name | 8,816 | Vendors, Products (SKU), LineItems, Amounts, Dates |
| EntityType_name | 8 | Invoice, Transaction, Vendor, Product, LineItem, Quantity, TotalAmount, Date |
| EdgeType_relationship_name | 13 | issued_by, paid_to, contains_item, refers_to, has_quantity, has_total, matches_to, etc. |
| TextDocument_name | 2,000 | Document references |
| TextSummary_text | 2,000 | Document summaries |

### New Data for Live Enrichment

`optional_data_for_enrichment/new_invoices.csv` — 8 invoices ready to ingest:
- Vendors 2, 3, 4, 9, 12, 15, 17, 20
- Products: monitors, SSDs, RAM, keyboards, laptops
- Total value: ~$70K
- Perfect for live demo of cognee pipeline

### Entity Extraction Schema

**Invoice entities**: INV-{vendor}-{month}-{id} → issued_by → Vendor, contains_item → LineItem → refers_to → Product (by SKU)

**Transaction entities**: TX-{vendor}-{month}-{id} → paid_to → Vendor, contains_item → LineItem

**Matching**: Invoice ↔ Transaction via shared Vendor + TotalAmount + Products

### Local Models

| Model | Role | Size | Via |
|---|---|---|---|
| cognee-distillabs-model-gguf-quantized | LLM reasoning | 2.5 GB | Ollama localhost:11434 |
| nomic-embed-text | Embeddings (768-dim) | 274 MB | Ollama localhost:11434 |
