"""
Resolve `LLM_BACKEND` to a `generate_answer`.

This module exists to retire a source edit. Switching between the hosted OpenAI
API and a local OpenAI-compatible server used to mean editing a commented-out
import in `app/workflow.py` — a code change to express a deployment choice,
which is not something a reader should have to diff to discover. Both backend
modules stay in the tree; only the selection moved into configuration.

`app/llm.py` and `app/llm_local.py` already expose an identical public surface
(`SYSTEM_PROMPT`, `generate_answer`), which is what makes this a rename rather
than an abstraction. There is no wrapper class and no protocol here on purpose:
the two functions are already interchangeable, and a layer that only forwards
would be one more thing to read.
"""

from app.config import settings

# Imported INSIDE the branch, not at module top. Both backend modules construct
# an OpenAI client at import time, so importing both would build a hosted-OpenAI
# client (and read OPENAI_API_KEY) on every local run, for a function that is
# never called. Lazy branching leaves the unused backend entirely unloaded.
if settings.llm_backend == "local":
    from app.llm_local import generate_answer
elif settings.llm_backend == "openai":
    from app.llm import generate_answer
else:
    # Loudly, at import, rather than falling back to a default. A typo in .env
    # that quietly changed which LLM you are billing for is the failure mode
    # this branch exists to prevent.
    raise ValueError(
        f"LLM_BACKEND must be 'local' or 'openai', got {settings.llm_backend!r}"
    )

__all__ = ["generate_answer"]
