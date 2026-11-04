"""Re-build hybrid index (local + optional Pinecone upsert)."""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.observability.logging import setup_logging
from backend.app.retrieval.hybrid import retriever


async def main():
    setup_logging()
    await retriever.initialize()
    print(f"Indexed {len(retriever.chunks)} chunks in mode={retriever.mode}")


if __name__ == "__main__":
    asyncio.run(main())
