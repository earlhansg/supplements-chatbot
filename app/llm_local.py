"""
Drop-in alternative to `app.llm` that talks to a local OpenAI-compatible
server (e.g. local-openai.exe on http://127.0.0.1:8080) instead of the
hosted OpenAI API.

Same public surface as `app.llm` — `SYSTEM_PROMPT` and `generate_answer` —
so switching is a one-line import change in `app/workflow.py`:

    from app.llm import generate_answer        # hosted OpenAI
    from app.llm_local import generate_answer  # local OpenAI-compatible server

The local server needs no real credentials; it accepts any api_key, so
LOCAL_LLM_API_KEY defaults to a placeholder. Run `GET /v1/models` on the
server to see which model IDs it accepts — unknown IDs come back as 404.
"""

from openai import OpenAI

from app.config import settings

# Local providers proxy to a slower backend than the hosted API, so allow a
# generous per-request timeout instead of the SDK's default.
_client = OpenAI(
    base_url=settings.local_llm_base_url,
    api_key=settings.local_llm_api_key,
    timeout=settings.local_llm_timeout_seconds,
)

SYSTEM_PROMPT = """You are a helpful customer support assistant for an online supplements store.

Answer the user's question using the FAQ context below when it's relevant. \
If the context doesn't cover the question, answer helpfully using general \
e-commerce knowledge, but stay on-topic for a supplements store. Keep answers \
concise (2-4 sentences) and friendly. Do not mention that you were given a \
context or FAQ list."""


def generate_answer(question: str, context: list[dict]) -> str:
    if context:
        context_block = "\n\n".join(
            f"Q: {item['question']}\nA: {item['answer']}" for item in context
        )
    else:
        context_block = "(no relevant FAQ found)"

    response = _client.chat.completions.create(
        model=settings.local_chat_model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"FAQ context:\n{context_block}\n\nCustomer question: {question}",
            },
        ],
    )
    return response.choices[0].message.content
