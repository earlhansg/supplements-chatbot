from openai import OpenAI

from app.config import settings

_client = OpenAI(api_key=settings.openai_api_key)

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
        model=settings.chat_model,
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
