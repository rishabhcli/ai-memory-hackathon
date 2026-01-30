# cognee-qdrant-starter

Hackathon starter repo for the AI-Memory Hackathon by cognee. Three FastAPI projects demonstrating semantic search, spend analytics, and anomaly detection on procurement data using Qdrant Cloud and local GGUF models.

## Stack

- Python 3.11+, FastAPI, uv for dependencies
- Qdrant Cloud (managed vector DB, 14,837 vectors, 768-dim, cosine distance)
- nomic-embed-text-v1.5 GGUF for embeddings (local, llama-cpp-python)
- Distil Labs SLM GGUF for reasoning/Q&A (local, Qwen3-4B fallback)

## Project Structure

```
project1-procurement-search/   # Semantic search UI — port 7777
project2-spend-analytics/      # Spend dashboard — port 5553
project3-anomaly-detective/    # Anomaly detection — port 6971
models/                        # Local GGUF models (gitignored)
snapshots/                     # Qdrant collection snapshots (gitignored)
restore-snapshots.py           # Restore snapshots to Qdrant Cloud
.github/skills/                # Agent skills for Qdrant patterns
```

Each project has: `app.py` (FastAPI + HTML UI), `main.py` (CLI/dev entry), `pyproject.toml`, `uv.lock`.

## Build & Run

```bash
cp .env.example .env   # add QDRANT_URL and QDRANT_API_KEY
uv run python restore-snapshots.py

cd project1-procurement-search
cp .env.example .env
uv sync
uv run python app.py
```

## Qdrant Collections

| Collection | Records | Content |
|---|---|---|
| DocumentChunk_text | 2,000 | Invoice and transaction chunks |
| Entity_name | 8,816 | Products, vendors, SKUs |
| EntityType_name | 8 | Entity type definitions |
| EdgeType_relationship_name | 13 | Relationship types |
| TextDocument_name | 2,000 | Document references |
| TextSummary_text | 2,000 | Document summaries |

## Code Style

- Use `uv` for all Python dependency management
- Use `.env` files for credentials, never hardcode secrets
- FastAPI for all web endpoints
- Inline HTML templates in `app.py` (no separate frontend build)
- Each project is fully self-contained

## Gotchas

- All collections use 768-dim vectors from nomic-embed-text. Do NOT mix embedding models in the same collection.
- Cloud inference (`cloud_inference=True`) uses a different model — requires re-embedding all data to use.
- Never commit `.env`, models, snapshots, or large binaries.
- Each project needs its own `.env` file with Qdrant credentials.

## Git Conventions

- Do not commit `.env`, models, snapshots, or large binaries
- Keep projects independent — changes to one project should not affect others

## Security

- All credentials in `.env` files (gitignored)
- No external API keys required — all models run locally
- Qdrant Cloud auth via API key in environment variables
