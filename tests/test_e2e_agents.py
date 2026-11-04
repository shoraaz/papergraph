"""
End-to-end: POST /api/query -> real Agno Team (leader + members, real
delegation and tool calls against the stores) -> event_mapper -> SSE.
Only the LLM is replaced, by tests/fake_llm.py.
"""

import dataclasses

import httpx
import pytest
from agno.models.openai.like import OpenAILike

import agents
import server
from tests.fake_llm import make_app
from tests.test_server import KEY, body, client, parse_sse  # noqa: F401,F811  (client is a fixture)


@pytest.fixture
def fake_llm(monkeypatch):
    app = make_app()

    def build(choice=None):
        return OpenAILike(
            id="fake-model",
            api_key="unused",
            base_url="http://fake-llm/v1",
            http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fake-llm"),
        )

    monkeypatch.setattr(agents, "_build_model", build)
    return app


async def test_real_team_run_streams_trace_answer_and_claims(client, fake_llm, monkeypatch):
    monkeypatch.setattr(server, "settings", dataclasses.replace(server.settings, mock_llm=False))
    r = await client.post("/api/query", json=body("Is Self-RAG's ChatGPT claim supported?"),
                          headers={"X-LLM-Api-Key": KEY})
    assert r.status_code == 200
    events = parse_sse(r.text)
    steps = [d for e, d in events if e == "step"]
    errors = [s for s in steps if s["kind"] == "error"]
    assert not errors, errors

    plans = [s["title"].lower() for s in steps if s["kind"] == "plan"]
    assert len(plans) == 3
    for title, member in zip(plans, ["retriever", "grader", "claimverifier"]):
        assert title.startswith("team leader delegates to") and member in title

    # Optional tool params are declared nullable (Groq rejects null vs "string").
    retriever_req = next(r for r in fake_llm.state.requests if any(
        t["function"]["name"] == "search_corpus" for t in r.get("tools") or []))
    schema = next(t["function"]["parameters"] for t in retriever_req["tools"] if t["function"]["name"] == "search_corpus")
    assert schema["properties"]["section"]["type"] == ["string", "null"]
    assert schema["properties"]["query"]["type"] == "string"

    retrieves = [s for s in steps if s["kind"] == "retrieve"]
    assert any("search_corpus" in s["title"] and "Retriever" in s["title"] for s in retrieves)
    verifier = [s for s in retrieves if "ClaimVerifier" in s["title"]]
    assert verifier and "section=results" in verifier[0]["query"]
    # Real retrieval through the tools: the fixture's Self-RAG results chunk.
    assert any("2310.11511v1 · results" in line for s in verifier for line in s["detail"])

    grades = [s for s in steps if s["kind"] == "grade"]
    assert grades and grades[0]["score"] == 0.85 and grades[0]["verdict"] == "sufficient"

    answer = next(d for e, d in events if e == "answer")
    assert "self-reflection tokens" in answer["answer"]
    assert answer["claims"] and answer["claims"][0]["verdict"] == "overstated"
    assert "".join(d["text"] for e, d in events if e == "delta").strip() == answer["answer"].strip()
    assert events[-1] == ("done", {"errored": False})
    assert server.state.gate.in_flight == 0


async def test_provider_error_is_surfaced_once_and_redacted(client, monkeypatch):
    monkeypatch.setattr(server, "settings", dataclasses.replace(server.settings, mock_llm=False))

    def build(choice=None):
        # A provider endpoint that rejects every call with the key echoed back.
        app = httpx.MockTransport(
            lambda req: httpx.Response(401, json={"error": {"message": f"Invalid key {KEY}"}})
        )
        return OpenAILike(id="fake", api_key=KEY, base_url="http://bad/v1",
                          http_client=httpx.AsyncClient(transport=app), max_retries=0)

    monkeypatch.setattr(agents, "_build_model", build)
    r = await client.post("/api/query", json=body(), headers={"X-LLM-Api-Key": KEY})
    events = parse_sse(r.text)
    errs = [d for e, d in events if e == "step" and d["kind"] == "error"]
    assert len(errs) == 1
    assert KEY not in r.text
    assert events[-1] == ("done", {"errored": True})
    assert server.state.gate.in_flight == 0
