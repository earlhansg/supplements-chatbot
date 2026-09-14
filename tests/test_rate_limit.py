"""
The 429 boundary and the Pydantic input bounds, through the real HTTP surface.

Kept last on purpose: instantiating `TestClient(app)` runs the lifespan, which
creates all three indexes and seeds both collections against the test instance.
That is useful — it exercises the bootstrap — but it makes this the slowest
file in the suite.

`app.workflow.generate_answer` is stubbed here too. A 429 test still has to let
`rate_limit_max_requests` real requests through, and each one is a full graph
invocation that would otherwise reach an LLM.
"""

import pytest
from fastapi.testclient import TestClient

from app import workflow
from app.config import settings
from app.main import app

from conftest import requires_redis

pytestmark = [requires_redis, pytest.mark.redis, pytest.mark.embeddings]

# Low enough that the assertions fit inside one fixed window. The window is
# `int(time.time()) // window_seconds`, so a test that fired 31 requests could
# straddle a boundary and watch the counter reset underneath it.
LIMIT = 3

ANSWER = "Standard shipping typically takes 3-5 business days within the continental US."


@pytest.fixture(autouse=True)
def _stub_llm(monkeypatch):
    monkeypatch.setattr(workflow, "generate_answer", lambda question, context: ANSWER)


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def small_limit(monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_max_requests", LIMIT)


def ask(client: TestClient, question: str = "How long does shipping take?"):
    return client.post("/chat", json={"question": question})


# --- the limit -----------------------------------------------------------------


def test_requests_up_to_the_limit_succeed(client, small_limit):
    for _ in range(LIMIT):
        assert ask(client).status_code == 200


def test_the_request_past_the_limit_is_a_429(client, small_limit):
    for _ in range(LIMIT):
        ask(client)

    response = ask(client)

    assert response.status_code == 429
    # A plain string, not a nested object, so the frontend's describeFailure()
    # in frontend/src/lib/backend.ts unpacks it as-is.
    assert isinstance(response.json()["detail"], str)
    assert int(response.headers["Retry-After"]) >= 1


def test_the_kill_switch_never_limits(client, small_limit, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_enabled", False)

    for _ in range(LIMIT + 3):
        assert ask(client).status_code == 200


def test_stats_is_deliberately_exempt(client, small_limit):
    """The limiter exists to bound LLM spend, not to ration a handful of Redis reads."""
    for _ in range(LIMIT + 1):
        ask(client)

    for _ in range(LIMIT + 3):
        assert client.get("/stats").status_code == 200


def test_health_is_not_limited(client, small_limit):
    for _ in range(LIMIT + 3):
        assert client.get("/health").status_code == 200


# --- the input bounds ----------------------------------------------------------


def test_an_empty_question_is_a_422(client):
    """`min_length=1` on ChatRequest. Mirrored at the Next.js boundary as a 400."""
    assert client.post("/chat", json={"question": ""}).status_code == 422


def test_an_over_length_question_is_a_422(client):
    too_long = "x" * (settings.max_question_chars + 1)

    assert client.post("/chat", json={"question": too_long}).status_code == 422


def test_a_question_at_the_length_limit_is_accepted(client):
    at_the_limit = "How long does shipping take? " + "x" * (settings.max_question_chars - 29)
    assert len(at_the_limit) == settings.max_question_chars

    assert client.post("/chat", json={"question": at_the_limit}).status_code == 200


def test_a_missing_question_field_is_a_422(client):
    assert client.post("/chat", json={}).status_code == 422


# --- the response contract -----------------------------------------------------


def test_chat_returns_every_documented_field(client):
    """`ChatResponse` is the wire contract frontend/src/lib/types.ts mirrors."""
    body = ask(client).json()

    assert set(body) == {
        "answer", "is_cached", "cache_similarity", "sources", "guardrail",
        "not_cached_reason", "cached_now", "cached_hits", "cache_band",
        "rejected_similarity",
    }


def test_stats_returns_every_documented_field(client):
    body = client.get("/stats").json()

    assert set(body) == {
        "hits", "misses", "blocked", "total", "hit_rate",
        "llm_calls_avoided", "cache_entries", "today",
    }
    assert set(body["today"]) == {"hits", "misses"}
