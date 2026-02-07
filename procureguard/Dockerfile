FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy dependency files first (for Docker layer caching)
COPY procureguard/pyproject.toml procureguard/.python-version procureguard/
COPY procureguard/uv.lock procureguard/ 2>/dev/null || true

# Copy shared modules and prompts
COPY shared/ shared/
COPY prompts/ prompts/

# Copy application code
COPY procureguard/app.py procureguard/agent.py procureguard/tools.py procureguard/graph_viz.py procureguard/

# Install dependencies
RUN cd procureguard && uv sync --no-dev

# Expose port
EXPOSE 8000

# Run the application
CMD ["sh", "-c", "cd procureguard && uv run python app.py"]
