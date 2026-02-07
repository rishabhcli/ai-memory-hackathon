# Setting Up Question Answering with Qdrant and a Local Model

## Quick Start

**We will set up**:
- Ollama with two local models (embedding and LLM)
- A Python virtual environment with pinned dependencies
- A Cognee knowledge graph imported from prebuilt data
- A local Qdrant vector store loaded with snapshot data
- The question answering script (`solution_q_and_a.py`)

**This will allow you to**:
- Access ingested data from invoice and transaction documents
- Retrieve structured context from a knowledge graph for LLM queries
- Ask natural-language questions about the data using a local language model
- Build tools, agents, or workflows on top of the Q&A pipeline

**Before installation**:
- copy `models/` from the USB to your working directory (or download via `uv run python download-from-spaces.py`)
- verify the three subdirectories contain Modelfile and a *.gguf each

**Project installation**:
```bash
# Ollama installation
brew install ollama
ollama serve

# Ollama model registration
cd models
ollama create nomic-embed-text -f nomic-embed-text/Modelfile
ollama create cognee-distillabs-model-gguf-quantized -f cognee-distillabs-model-gguf-quantized/Modelfile
cd ..

# Initialize python environment, install dependencies
uv venv
source .venv/bin/activate
uv sync

# Graph setup
python setup.py

# Qdrant (local Docker)
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant

# Configure for use locally, retrieve data, restore to database
cp .env.example.local .env
uv run python download-from-spaces.py
uv run python restore-snapshots.py

# Run Q and A example
python solution_q_and_a.py
```

**Pitfalls to avoid**:
- building the venv in `models/` instead of the project root
- having a stale venv activated

**Next steps**:
- look around the code
- play with the queries
- check out the databases
- build something


## Useful Code

Turn off and remove qdrant if necessary for recreating:
```
# Stop and remove the container
docker stop qdrant && docker rm qdrant

# Remove the persistent volume
docker volume rm qdrant_storage

``