# Setting Up Question Answering with Qdrant and a Local Model

This guide explains how to set up question answering using the cognee-distillabs local model with Ollama.

## Prerequisites

### 1. Install Ollama

Install Ollama from [ollama.com/](https://ollama.com/) or using your package manager.

### 2. Register Ollama Model

First, copy over the models directory from the USB stick into the hackathon directory. Then, navigate to the models directory containing the downloaded model files and the register the models with:

```bash
cd models

ollama create nomic-embed-text -f nomic-embed-text/Modelfile
ollama create cognee-distillabs-model-gguf-quantized -f cognee-distillabs-model-gguf-quantized/Modelfile

cd ..
```

### 3. Graph Setup

You will need a virtual environment (venv) configured for this, so if you didn't create one beforehand, here is how you can do it:

```bash
# You can easily do it with the help of uv
uv venv

# In case you don't have uv installed, you can do the following command
python -m venv venv

# After this, you can activate the environment
source .venv/bin/activate
```

In order to import the generated graph into your venv memory, you should run the following script:

```bash
uv pip install cognee transformers
python setup.py
```

or if you do not like uv (you should like it), use 

```bash
python -m pip install cognee transformers
python setup.py
```

### 4. Qdrant (Vector) Setup

First, install the Qdrant adapter in your venv:

```bash
uv pip install qdrant-client cognee-community-vector-adapter-qdrant
```

Then, clone the [cognee-qdrant-starter repo](https://github.com/thierrypdamiba/cognee-qdrant-starter) into your hackathon directory. Now choose **one** of the two options below.

---

#### Option A: Qdrant Cloud (Hosted)

Best for: deployment, sharing with teammates, no Docker needed.

**A1. Create a free Qdrant Cloud cluster**

1. Go to [https://cloud.qdrant.io](https://cloud.qdrant.io) and sign up
2. Create a new cluster (free tier is fine)
3. Copy your Cluster URL and API Key

**A2. Configure credentials, download, and restore**

```bash
cd cognee-qdrant-starter

# Create your .env with Qdrant credentials
cp .env.example .env
# Edit .env — fill in QDRANT_URL and QDRANT_API_KEY with your Cloud values

# Download the 6 snapshot files (~91 MB total)
uv run python download-from-spaces.py

# Upload them to your Qdrant Cloud cluster
uv run python restore-snapshots.py

cd ..
```

After this, your Qdrant Cloud cluster contains 14,837 vectors and `solution_q_and_a.py` will automatically read your credentials from this `.env`.

---

#### Option B: Qdrant Local (Docker)

Best for: offline development, no cloud account needed, full control.

*B1. Start a local Qdrant instance*

```bash
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

This gives you Qdrant running at `http://localhost:6333`. No API key needed for local.

*B2. Configure credentials, download, and restore*

```bash
cd cognee-qdrant-starter
cp .env.example .env
```

Edit `.env` — set the URL to your local instance:

```
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
```

```bash
uv run python download-from-spaces.py   # download snapshot files
uv run python restore-snapshots.py       # restore to local Qdrant

cd ..
```

After this, your local Qdrant instance contains 14,837 vectors and `solution_q_and_a.py` will automatically read your credentials from this `.env`.

### 5. Running the Script

Once configured, run the question answering script:

```bash
python solution_q_and_a.py
```

### 6. Example Results

Example results comparing LLM and SLM outputs can be found in `responses.txt`.



## Appendix 

### Appendix A: Example Projects using Qdrant

#### Project 1: Procurement Semantic Search (port 7777)

Semantic search across all procurement data with interactive UI.

**Qdrant features:** Query API, Prefetch + RRF Fusion, Group API, Discovery API, Recommend API, payload indexing, filtered search

**Endpoints:** `/search`, `/search/grouped`, `/discover`, `/recommend`, `/filter`, `/ask` (RAG Q&A), `/cognee-search`, `/add-knowledge`, `/collections`

#### Project 2: Spend Analytics Dashboard (port 5553)

Interactive analytics dashboard with Chart.js visualizations and semantic search.

**Qdrant features:** Scroll API (bulk extraction), Query API, Group API, payload indexing

**Endpoints:** `/api/analytics`, `/api/search`, `/api/search/grouped`, `/api/insights` (LLM analysis)

#### Project 3: Anomaly Detective (port 6971)

Automated anomaly detection using vector analysis and Qdrant's batch API.

**Qdrant features:** Batch Query API (50 recommend queries/request), Recommend API, Scroll API with vectors, payload indexing

**Detection methods:** amount outliers (z-score), embedding outliers (centroid distance), near-duplicates (similarity > 0.99), vendor variance

**Endpoints:** `/api/anomalies`, `/api/search`, `/api/investigate/{point_id}`, `/api/explain/{point_id}` (LLM explanation)


### Appendix B: DigitalOcean Deployment

#### Deployment

Two modes: **local** (dev with GGUF models) and **remote** (deployed with API-based inference).

### Local dev (Distil Labs GGUF)

```bash
# .env: LLM_MODE=local, EMBED_MODE=local (defaults)
uv run python app.py
```

Runs the Distil Labs SLM locally via llama-cpp-python. Requires 4-8GB RAM.

### Deploy to DigitalOcean App Platform

```bash
# 1. Upload data to DO Spaces
uv run python upload-to-spaces.py

# 2. Set remote mode in .env
#    LLM_MODE=remote
#    LLM_API_URL=<distil-labs-hosted-endpoint>
#    EMBED_MODE=remote
#    EMBED_API_URL=<embedding-api-endpoint>

# 3. Deploy
doctl apps create --spec .do/app.yaml
```

Or use Docker locally:

```bash
docker compose up
```

The deployed version calls the Distil Labs hosted API (or any OpenAI-compatible endpoint) instead of loading GGUF files. This keeps the container small and runs on a $6/mo App Platform instance.

**Free credits:** New DigitalOcean accounts get [$200 in free credits](https://www.digitalocean.com/try/free-trial) for 60 days.

#### Environment variables


| Variable          | Default           | Description                                                     |
| ----------------- | ----------------- | --------------------------------------------------------------- |
| `QDRANT_URL`      | -                 | Qdrant Cloud cluster URL                                        |
| `QDRANT_API_KEY`  | -                 | Qdrant Cloud API key                                            |
| `LLM_MODE`        | `local`           | `local` (GGUF) or `remote` (API)                                |
| `LLM_API_URL`     | -                 | OpenAI-compatible chat completions endpoint                     |
| `LLM_API_KEY`     | -                 | API key for remote LLM                                          |
| `LLM_MODEL_NAME`  | `distil-labs-slm` | Model name for remote LLM                                       |
| `EMBED_MODE`      | `local`           | `local` (GGUF) or `remote` (API)                                |
| `EMBED_API_URL`   | -                 | OpenAI-compatible embeddings endpoint                           |
| `EMBED_API_KEY`   | -                 | API key for remote embeddings                                   |
| `SPACES_ENDPOINT` | -                 | DO Spaces endpoint (e.g. `https://nyc3.digitaloceanspaces.com`) |
| `SPACES_BUCKET`   | -                 | DO Spaces bucket name                                           |



### Appendix C: Adding your own data

The starter data was built using cognee's ECL (Extract, Cognify, Load) pipeline. You can add your own data:

#### Quick start

```bash
cd cognee-qdrant-starter/cognee-pipeline
cp .env.example .env
# Edit .env: add Qdrant credentials + LLM provider
uv sync
uv run python ingest.py
```

#### Add your own data

```python
import cognee
from cognee.api.v1.search import SearchType

# 1. Add documents (text, files, URLs)
await cognee.add("Your document text here...")
await cognee.add("/path/to/document.pdf")
await cognee.add(["doc1.txt", "doc2.csv", "doc3.pdf"])

# 2. Build knowledge graph (extracts entities, relationships, summaries)
await cognee.cognify()

# 3. Search with graph context
results = await cognee.search(
    query_text="What vendors supply IT equipment?",
    query_type=SearchType.CHUNKS,  # or SUMMARIES, GRAPH_COMPLETION, RAG_COMPLETION
)
```

#### Supported input types

- Plain text strings
- PDF, DOCX, TXT, CSV files
- URLs (web pages)
- Directories of files

See [cognee docs](https://docs.cognee.ai) for full pipeline options.


### Appendix D: Using base QWEN3 model with ollama

To use the base QWEN3 model, Navigate to the models directory containing the downloaded model files and the register the models with:

```bash
cd models

ollama create Qwen3-4B-Q4_K_M -f Qwen3-4B-Q4_K_M/Modelfile
```

Once its loaded, you can use it using standard [OpenAI interface](https://ollama.com/blog/openai-compatibility). For example from python as

```python
from openai import OpenAI

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

response = client.chat.completions.create(
  model="Qwen3-4B-Q4_K_M",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The LA Dodgers won in 2020."},
    {"role": "user", "content": "Where was it played?"}
  ]
)
print(response.choices[0].message.content)
```

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- [cognee](https://github.com/topoteretes/cognee) (knowledge graph memory)
- [Qdrant Cloud](https://cloud.qdrant.io) cluster (free tier available)
- [DigitalOcean](https://www.digitalocean.com/) account ($200 free credits available)
