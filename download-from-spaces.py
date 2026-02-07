"""
Download snapshots and models from DigitalOcean Spaces (S3-compatible).
Falls back to the original S3 URLs if SPACES_ENDPOINT is not set.

Usage:
    cp .env.example .env  # configure Spaces or use S3 defaults
    uv run python download-from-spaces.py
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

SPACES_ENDPOINT = os.getenv("SPACES_ENDPOINT")
SPACES_BUCKET = os.getenv("SPACES_BUCKET")

# S3 fallback URLs
S3_BASE = "https://cognee-qdrant-starter.s3.amazonaws.com"

SNAPSHOT_FILES = [
    "snapshots/DocumentChunk_text-6835894903267623-2026-01-30-01-16-23.snapshot",
    "snapshots/Entity_name-6835894903267623-2026-01-30-01-16-24.snapshot",
    "snapshots/EntityType_name-6835894903267623-2026-01-30-01-16-24.snapshot",
    "snapshots/EdgeType_relationship_name-6835894903267623-2026-01-30-01-16-25.snapshot",
    "snapshots/TextDocument_name-6835894903267623-2026-01-30-01-16-25.snapshot",
    "snapshots/TextSummary_text-6835894903267623-2026-01-30-01-16-25.snapshot",
]

MODEL_FILE = "models.zip"


def get_url(path: str) -> str:
    if SPACES_ENDPOINT and SPACES_BUCKET:
        return f"{SPACES_ENDPOINT}/{SPACES_BUCKET}/{path}"
    return f"{S3_BASE}/{path}"


def download_file(url: str, dest: str):
    dirname = os.path.dirname(dest)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return
    print(f"  Downloading {url}...")
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    size_mb = os.path.getsize(dest) / 1e6
    print(f"  Saved {dest} ({size_mb:.1f}MB)")


def main():
    source = f"DO Spaces ({SPACES_ENDPOINT}/{SPACES_BUCKET})" if (SPACES_ENDPOINT and SPACES_BUCKET) else f"S3 ({S3_BASE})"
    print(f"Downloading from {source}\n")

    print("Downloading snapshots...")
    for path in SNAPSHOT_FILES:
        download_file(get_url(path), path)

    print(f"\nDownloading models...")
    download_file(get_url(MODEL_FILE), MODEL_FILE)

    print("\nDone. Next steps:")
    print("  1. unzip models.zip -d models/")
    print("  2. uv run python restore-snapshots.py")


if __name__ == "__main__":
    main()
