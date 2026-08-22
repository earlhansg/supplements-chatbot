# Supplements Store Chatbot (backend-only)

A FastAPI + LangGraph + OpenAI backend (with local, open-source embeddings)
for an e-commerce support chatbot,
built as a Python translation of the patterns used in the local
`reference/redish/openai-version` project (RedisJSON documents + a
RediSearch vector index for KNN lookups, and a LangGraph workflow that
checks a cache before ever calling the LLM).

The reference project's semantic cache uses Redis's managed **LangCache**
API. This project doesn't depend on that managed service — it implements
the same "embed the question, KNN search, threshold on similarity" idea
directly against a self-hosted Redis Stack instance, so the whole thing
runs with just `docker compose up`.

## Tech stack

- **FastAPI** — HTTP API, served docs at `/docs`
- **LangGraph** — orchestrates: check cache → (miss) retrieve context → generate answer → save to cache
- **OpenAI** — `gpt-4o-mini` for chat
- **sentence-transformers** — `BAAI/bge-base-en-v1.5` (local, open-source, no API cost) for embeddings
- **Redis Stack** (RedisJSON + RediSearch) — knowledge base storage, vector search, and semantic cache
- **Pydantic** — request/response schemas (`app/schemas.py`) and settings (`app/config.py`)

## Redis data model

Two RedisJSON collections, each with its own RediSearch vector index.

### 1. Knowledge base — `kb:<id>`

```json
// key: kb:faq-005
{
  "id": "faq-005",
  "category": "shipping",
  "question": "How much does shipping cost?",
  "answer": "Standard shipping is free on all orders over $50. ...",
  "embedding": [0.0123, -0.0456, ...]   // 768 floats, BAAI/bge-base-en-v1.5
}
```

Index `idx:kb` (`FT.CREATE idx:kb ON JSON PREFIX 1 kb: SCHEMA ...`):

| JSON path       | Alias       | Type              |
|------------------|-------------|-------------------|
| `$.question`     | `question`  | TEXT              |
| `$.answer`       | `answer`    | TEXT              |
| `$.category`     | `category`  | TAG               |
| `$.embedding`    | `embedding` | VECTOR (HNSW, COSINE, DIM 768, FLOAT32) |

On every question, the workflow embeds the question and runs a `KNN 3`
query against `idx:kb` to pull the 3 most relevant FAQs as context for the LLM.

### 2. Semantic cache — `cache:<uuid>`

```json
// key: cache:1f2e3d4c-...
{
  "query": "how long till my refund shows up",
  "answer": "Once we receive and inspect your return, refunds are processed within 3-5 business days...",
  "embedding": [0.0231, -0.0198, ...]
}
```

Index `idx:cache` (same shape as `idx:kb`, over the `cache:` prefix):

| JSON path     | Alias    | Type                                      |
|----------------|----------|--------------------------------------------|
| `$.query`      | `query`  | TEXT                                       |
| `$.answer`     | `answer` | TEXT                                       |
| `$.embedding`  | `embedding` | VECTOR (HNSW, COSINE, DIM 768, FLOAT32) |

Every `cache:*` key gets a Redis `EXPIRE` set to `CACHE_TTL_SECONDS`
(default 24h), so entries age out on their own — no separate cleanup job.
Because RediSearch keeps its index in sync with keyspace expirations, an
expired entry simply stops showing up in KNN results.

**Cache lookup logic:** embed the incoming question, run `KNN 1` against
`idx:cache`, convert the returned cosine distance to a similarity
(`1 - score`), and treat it as a hit only if `similarity >=
CACHE_SIMILARITY_THRESHOLD` (default `0.78`, calibrated for
`BAAI/bge-base-en-v1.5` — its cosine similarities run lower than OpenAI's for
same-topic paraphrases, so this threshold is model-dependent; re-tune it if
you swap embedding models). This is what makes it a
*semantic* cache — "how long till my refund shows up" can hit a cache
entry saved for "How long does it take to get my refund?" even though the
wording differs.

## Workflow (LangGraph)

```
START -> check_cache --(hit)--> END
             |
          (miss)
             v
      retrieve_context -> generate_answer -> save_cache -> END
```

1. **check_cache** — embed the question, KNN search `idx:cache`. On a hit,
   set `is_cached=True`, log `CACHE HIT`, and route straight to `END`.
2. **retrieve_context** (miss only) — KNN search `idx:kb` for the top-K
   relevant FAQs.
3. **generate_answer** — call the LLM with the FAQ context + question.
4. **save_cache** — embed the question again and store `{query, answer,
   embedding}` under a fresh `cache:<uuid>` key with a TTL. Logs `CACHE
   MISS / NOT CACHED`.

The API response always includes `is_cached: true/false` (see
`app/schemas.py::ChatResponse`).

## Getting started

### 1. Start Redis Stack

```bash
docker compose up -d
```

This runs `redis/redis-stack`, which bundles the RedisJSON and
RediSearch modules that `FT.CREATE` / `FT.SEARCH` and `JSON.SET` need
(plain `redis:latest` does **not** include these modules), plus the
**RedisInsight** web UI on port `8001` — open
**http://localhost:8001/redis-stack/browser** to browse keys, run
commands, and inspect the `kb:*` / `cache:*` documents visually.

### 2. Configure environment

```bash
cp .env.example .env
# then edit .env and set OPENAI_API_KEY (still used for chat completions)
```

### 3. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

`sentence-transformers` pulls in PyTorch, so this install is heavier than
before. The `BAAI/bge-base-en-v1.5` embedding model (~440MB) is downloaded
from Hugging Face on first run and cached locally
(`~/.cache/huggingface`/`%USERPROFILE%\.cache\huggingface` on Windows) — no
API key or network access is needed for embeddings after that.

### 4. Run the API

```bash
uvicorn app.main:app --reload
```

On startup the app creates both RediSearch indexes and, if the knowledge
base is empty, automatically embeds and loads the 10 sample FAQs from
`data/faqs.json`. To reseed manually instead (e.g. after editing the FAQ
file), run:

```bash
python scripts/load_kb.py
```

## Running it

```bash
uvicorn app.main:app --reload
```

Keep this running in its own terminal — it prints `CACHE HIT` /
`CACHE MISS / NOT CACHED` for every request, which is the easiest way to
watch the workflow's routing decision live while you test.

## Testing it

### Option A — Swagger UI (`/docs`)

1. Open **http://127.0.0.1:8000/docs**.
2. Expand `POST /chat` → **Try it out**.
3. Send a body like:
   ```json
   { "question": "How long does shipping take?" }
   ```
4. Check the response: `is_cached` should be `false`, `cache_similarity`
   should be `null`, and `sources` should list 3 FAQs pulled from the
   knowledge base. The server console should log `CACHE MISS / NOT CACHED`.
5. Send the **exact same** question again. This time `is_cached` should be
   `true`, `cache_similarity` should be close to `1.0`, `sources` should be
   empty (the cache hit skips KB retrieval entirely), and the console
   should log `CACHE HIT`.
6. Send a **reworded** version of the same question, e.g.
   `"how many days till my package arrives"`. Because the cache match is
   semantic (embedding similarity), this should also come back as a cache
   hit even though no word overlaps with the original question.
7. Try `GET /health` — should return `{"status": "ok"}`.

### Option B — curl

```bash
# health check
curl http://127.0.0.1:8000/health

# first call -> cache miss, hits the LLM + knowledge base
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is your refund policy?"}'

# same question again -> cache hit, no LLM/KB call
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is your refund policy?"}'

# reworded question -> still a cache hit (semantic match)
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "when do I get my money back"}'
```

Other good questions to try, one per FAQ category in `data/faqs.json`:
`"My order arrived damaged, can I get a replacement?"`,
`"How much does shipping cost?"`,
`"How can I track my order?"`,
`"Can I cancel my order after placing it?"`,
`"Are your supplements third-party tested?"`.
Also try something the FAQs don't cover (e.g. `"Do you ship to the moon?"`)
to confirm the bot still answers sensibly using general knowledge instead
of erroring.

### Option C — inspect Redis directly

**Via RedisInsight (web UI):** open
**http://localhost:8001/redis-stack/browser**, connect to the local
Redis instance (host `redis`/`localhost`, port `6379`, no auth), and
browse the `kb:*` / `cache:*` keys, run `FT.SEARCH` / `JSON.GET` from
the built-in CLI, and watch TTLs count down on cache entries — all
without leaving the browser.

**Via `redis-cli`** (or `docker compose exec redis redis-cli`), you can watch
the two indexes and collections the app is reading/writing:

```bash
# how many FAQs / cache entries exist right now
FT.SEARCH idx:kb "*" LIMIT 0 0
FT.SEARCH idx:cache "*" LIMIT 0 0

# look at one FAQ document
JSON.GET kb:faq-001

# after asking a question via curl/Swagger, list the cache entry it created
KEYS cache:*
JSON.GET cache:<the-uuid-you-got-back> $.query $.answer

# confirm the cache entry has a TTL (in seconds)
TTL cache:<the-uuid-you-got-back>
```

### Resetting state between test runs

```bash
# wipe and reseed just the knowledge base + cache indexes/data
docker compose down -v && docker compose up -d
python scripts/load_kb.py
```

**Note:** if you're migrating an existing Redis instance from the OpenAI
embeddings (1536-dim) to the local model (768-dim), you must wipe the
volume as above — `FT.CREATE` won't alter an existing index's vector
dimension, so old and new embeddings can't coexist in the same index.

## Project layout

```
supplements-chatbot/
├── app/
│   ├── config.py          # pydantic-settings (.env)
│   ├── schemas.py          # ChatRequest/ChatResponse pydantic models
│   ├── redis_client.py     # shared redis-py connection
│   ├── vector_utils.py     # float list <-> FLOAT32 bytes
│   ├── embeddings.py       # local sentence-transformers embeddings wrapper
│   ├── knowledge_base.py   # kb:* documents + idx:kb (RediSearch)
│   ├── semantic_cache.py   # cache:* documents + idx:cache (RediSearch)
│   ├── llm.py               # OpenAI chat completion wrapper
│   ├── workflow.py          # LangGraph StateGraph
│   └── main.py               # FastAPI app + /chat endpoint
├── data/faqs.json           # 10 sample FAQs (knowledge base seed data)
├── scripts/load_kb.py        # standalone KB loader
├── docker-compose.yml         # redis-stack (+ RedisInsight UI on :8001)
└── .env.example
```
