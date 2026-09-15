"""
Standalone knowledge-base loader, mirroring the reference project's
`scripts/load-products.js`. Useful for re-seeding the FAQ knowledge base and
the guardrail exemplars without starting the API server.

Unlike the API's startup bootstrap, this always reloads both collections — so
it is the way to pick up an edit to an existing FAQ or exemplar's *text*,
which leaves the document count unchanged and therefore does not trigger the
lifespan re-seed.

Usage:
    python scripts/load_kb.py [path/to/faqs.json [path/to/guardrail_examples.json]]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.guardrails import create_guardrail_index, load_guardrail_examples  # noqa: E402
from app.knowledge_base import create_kb_index, load_faqs  # noqa: E402
from app.semantic_cache import create_cache_index  # noqa: E402

DEFAULT_FAQS_PATH = Path(__file__).resolve().parent.parent / "data" / "faqs.json"
DEFAULT_GUARDRAIL_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "guardrail_examples.json"
)


def main():
    faqs_path = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_FAQS_PATH)
    guardrail_path = sys.argv[2] if len(sys.argv) > 2 else str(DEFAULT_GUARDRAIL_PATH)

    print("Creating RediSearch indexes...")
    create_kb_index()
    create_cache_index()
    create_guardrail_index()

    print(f"Loading FAQs from: {faqs_path}")
    count = load_faqs(faqs_path)
    print(f"Loaded {count} FAQs into Redis")

    print(f"Loading guardrail exemplars from: {guardrail_path}")
    count = load_guardrail_examples(guardrail_path)
    print(f"Loaded {count} guardrail exemplars into Redis")


if __name__ == "__main__":
    main()
