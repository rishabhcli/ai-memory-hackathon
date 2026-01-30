![AI-Memory Hackathon by cognee](hackathon-banner.avif)

# cognee-qdrant-starter

Starter templates for the **AI-Memory Hackathon by cognee** — three ready-to-run FastAPI projects that demonstrate semantic search, analytics, and anomaly detection on procurement data using **local models** (Distil Labs SLM + nomic-embed-text) and **Qdrant Cloud**.

## Quick Start

### 1. Set up Qdrant Cloud

Create a free cluster at [cloud.qdrant.io](https://cloud.qdrant.io), then download and restore the provided snapshots:

```bash
# Download snapshots from S3
mkdir -p snapshots
curl -O --output-dir snapshots https://cognee-qdrant-starter.s3.amazonaws.com/snapshots/DocumentChunk_text-6835894903267623-2026-01-30-01-16-23.snapshot
curl -O --output-dir snapshots https://cognee-qdrant-starter.s3.amazonaws.com/snapshots/Entity_name-6835894903267623-2026-01-30-01-16-24.snapshot
curl -O --output-dir snapshots https://cognee-qdrant-starter.s3.amazonaws.com/snapshots/EntityType_name-6835894903267623-2026-01-30-01-16-24.snapshot
curl -O --output-dir snapshots https://cognee-qdrant-starter.s3.amazonaws.com/snapshots/EdgeType_relationship_name-6835894903267623-2026-01-30-01-16-25.snapshot
curl -O --output-dir snapshots https://cognee-qdrant-starter.s3.amazonaws.com/snapshots/TextDocument_name-6835894903267623-2026-01-30-01-16-25.snapshot
curl -O --output-dir snapshots https://cognee-qdrant-starter.s3.amazonaws.com/snapshots/TextSummary_text-6835894903267623-2026-01-30-01-16-25.snapshot

# Add your credentials
cp .env.example .env
# Edit .env with your Qdrant Cloud URL and API key

# Restore all 6 collections from snapshots
uv run python restore-snapshots.py
```

This uploads 6 pre-built collections (14,837 vectors, 768-dim, with payload indexes) to your cluster:

| Collection | Records | Content |
|---|---|---|
| DocumentChunk_text | 2,000 | Invoice and transaction chunks |
| Entity_name | 8,816 | Products, vendors, SKUs |
| EntityType_name | 8 | Entity type definitions |
| EdgeType_relationship_name | 13 | Relationship types |
| TextDocument_name | 2,000 | Document references |
| TextSummary_text | 2,000 | Document summaries |

### 2. Download and place the models

Download the models zip from S3 and extract:

```bash
curl -O https://cognee-qdrant-starter.s3.amazonaws.com/models.zip
unzip models.zip -d models/
```

This gives you:

```
models/
  nomic-embed-text/
    nomic-embed-text-v1.5.f16.gguf          # embeddings
  cognee-distillabs-model-gguf-quantized/
    model-quantized.gguf                      # Distil Labs SLM for reasoning
  Qwen3-4B-Q4_K_M/
    Qwen3-4B-Q4_K_M.gguf                     # Qwen3 fallback (optional)
```

The Distil Labs model powers the RAG Q&A and anomaly explanation endpoints. If not found, Qwen3-4B is used as fallback.

### 3. Run a project

Each project is self-contained:

```bash
cd project1-procurement-search  # or project2 or project3
cp .env.example .env
# Edit .env with your Qdrant Cloud URL and API key

uv sync
uv run python app.py
```

## Projects

### Project 1: Procurement Semantic Search (port 7777)

Semantic search across all procurement data with interactive UI.

**Qdrant features:**
- **Query API** — vector similarity search with local embeddings
- **Group API** (`query_points_groups`) — group results by payload field
- **Point-ID queries** — "More like this" / "Less like this" discovery
- **Payload indexing** (`create_payload_index`) — keyword + full-text indexes
- **Filtered search** — combine vector similarity with metadata filters

**Endpoints:** `/search`, `/search/grouped`, `/discover`, `/recommend`, `/filter`, `/ask` (RAG Q&A), `/collections`

---

### Project 2: Spend Analytics Dashboard (port 5553)

Interactive analytics dashboard with Chart.js visualizations and semantic search.

**Qdrant features:**
- **Scroll API** — bulk data extraction for aggregation
- **Query API** — semantic search over invoices
- **Group API** — vendor-grouped search results
- **Payload indexing** — fast vendor/type filtering

**Shows:** $13.4M total spend, 1000 invoices + 1000 transactions, 10 vendors, monthly trends, top products by qty/revenue.

**Endpoints:** `/api/analytics`, `/api/search`, `/api/search/grouped`, `/api/insights` (LLM analysis)

---

### Project 3: Anomaly Detective (port 6971)

Automated anomaly detection using vector analysis and Qdrant's batch API.

**Qdrant features:**
- **Batch Query API** (`query_batch_points`) — 50 recommend queries per request
- **Point-ID queries** — find records similar to flagged anomalies
- **Scroll API** with vectors — bulk vector retrieval for centroid analysis
- **Payload indexing** — fast anomaly filtering

**Detection methods:**
- Amount outliers (z-score > 2.5)
- Embedding outliers (distance from centroid, z > 2.0)
- Near-duplicates (similarity > 0.99 via batch recommend)
- Vendor variance (coefficient of variation > 0.8)

**Endpoints:** `/api/anomalies`, `/api/search`, `/api/investigate/{point_id}`, `/api/explain/{point_id}` (LLM explanation)

## Qdrant Features Matrix

| Feature | P1 | P2 | P3 |
|---|:---:|:---:|:---:|
| Query API (vector search) | x | x | x |
| Batch Query API | | | x |
| Point-ID Recommend | x | | x |
| Group API | x | x | |
| Scroll API | | x | x |
| Payload Indexing | x | x | x |
| Filtered Search | x | x | |

## Architecture

```
User Query
    |
    v
+---------------------+
|  nomic-embed-text    |  <-- Local GGUF (768-dim embeddings)
|  (llama-cpp-python)  |
+---------+-----------+
          | query vector
          v
+---------------------+
|   Qdrant Cloud       |  <-- 14,837 vectors, 6 collections
|   (managed cluster)  |
+---------+-----------+
          | results
          v
+---------------------+
|  Distil Labs SLM     |  <-- Local GGUF (reasoning, Q&A, explanation)
|  (llama-cpp-python)  |
+---------+-----------+
          | answer
          v
+---------------------+
|   FastAPI App        |  <-- RAG Q&A, charts, anomaly detection
+---------------------+
```

All models run locally via `llama-cpp-python` — no external API keys needed.

## Data

IT hardware procurement records:
- Invoices and transactions with line items (laptops, monitors, keyboards, SSDs, RAM)
- 10 vendors, ~2000 documents, spanning 2025
- Entities extracted by Cognee (product names, SKUs, vendor IDs)

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- [Qdrant Cloud](https://cloud.qdrant.io) cluster (free tier works)
