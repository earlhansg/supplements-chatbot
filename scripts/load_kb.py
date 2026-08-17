"""
Standalone knowledge-base loader, mirroring the reference project's
`scripts/load-products.js`. Useful for re-seeding the FAQ knowledge base
without starting the API server.

Usage:
    python scripts/load_kb.py [path/to/faqs.json]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.knowledge_base import create_kb_index, load_faqs  # noqa: E402
from app.semantic_cache import create_cache_index  # noqa: E402

DEFAULT_FAQS_PATH = Path(__file__).resolve().parent.parent / "data" / "faqs.json"


def main():
    faqs_path = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_FAQS_PATH)

    print("Creating RediSearch indexes...")
    create_kb_index()
    create_cache_index()

    print(f"Loading FAQs from: {faqs_path}")
    count = load_faqs(faqs_path)
    print(f"Loaded {count} FAQs into Redis")


if __name__ == "__main__":
    main()
