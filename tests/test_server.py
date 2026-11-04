"""API behaviour: BYOK gating, rate limits, load shedding, streaming, isolation."""

import asyncio
import dataclasses
import json
import time

import httpx
import pytest
from qdrant_client import AsyncQdrantClient

import server
from embeddings import HashingEmbedder
from graph_view import build_graph_view
from ratelimit import ConcurrencyGate, Limit, MemoryLimiter, RateLimiter
from security import redact
from stores import Stores
from stores.qdrant_store import QdrantVectorStore
from stores.snapshot_graph import SnapshotGraphStore
from tests.fixtures import CHUNKS, SHARED
from tests.test_graph_contract import fixture_snapshot

KEY = "gsk_" + "a" * 40
SESSION = "sess_test_000001"


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for frame in body.replace("\r\n", "\n").split("\n\n"):
        ev, data = "message", ""
        for line in frame.split("\n"):
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        if data:
            events.append((ev, json.loads(data)))
    return events


@pytest.fixture
async def client(monkeypatch):
    v = QdrantVectorStore(AsyncQdrantClient(location=":memory:"), "t", HashingEmbedder(384), embedded=True)
    await v.ensure_ready()
    await v.upsert_chunks(CHUNKS)
    stores = Stores(v, SnapshotGraphStore(fixture_snapshot()), "demo", {"papers": 4, "chunks": len(CHUNKS)})
    monkeypatch.setattr(server.state, "stores", stores)
    monkeypatch.setattr(server.state, "shared_view", await build_graph_view(stores.graph, [SHARED]))
    monkeypatch.setattr(server.state, "limiter", RateLimiter(MemoryLimiter()))
    monkeypatch.setattr(server.state, "gate", ConcurrencyGate(40, 2))

    async def key_ok(provider, key):
        return True

    monkeypatch.setattr(server.state.keys, "check", key_ok)
    monkeypatch.setattr(server, "settings", dataclasses.replace(server.settings, mock_llm=True))
    monkeypatch.setenv("MOCK_LATENCY_S", "0.05")
    import mock_run

    monkeypatch.setattr(mock_run, "LATENCY_S", 0.05)
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as c:
        yield c
    await stores.close()


def body(message="What does Self-RAG do?", **kw):
    return {"message": message, "session_id": SESSION, "model": {"provider": "groq"}, **kw}


async def test_query_requires_visitor_key(client):
    r = await client.post("/api/query", json=body())
    assert r.status_code == 401
    assert "API key" in r.json()["detail"]


async def test_invalid_key_rejected_before_run(client, monkeypatch):
    monkeypatch.setattr(server, "settings", dataclasses.replace(server.settings, mock_llm=False))

    async def key_bad(provider, key):
        return False

    monkeypatch.setattr(server.state.keys, "check", key_bad)
    r = await client.post("/api/query", json=body(), headers={"X-LLM-Api-Key": KEY})
    assert r.status_code == 401 and "rejected" in r.json()["detail"]
    assert server.state.gate.in_flight == 0


async def test_mock_run_streams_and_releases_slot(client):
    r = await client.post("/api/query", json=body(), headers={"X-LLM-Api-Key": KEY})
    assert r.status_code == 200
    events = parse_sse(r.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "step" and "delta" in kinds and kinds[-2:] == ["answer", "done"]
    retrieve = [d for e, d in events if e == "step" and d["kind"] == "retrieve"]
    assert any("search_corpus" in d["title"] for d in retrieve)
    assert server.state.gate.in_flight == 0
    assert r.headers["x-ratelimit-limit"]


async def test_rate_limit_returns_429_with_retry_after(client, monkeypatch):
    monkeypatch.setattr(server, "QUERY_LIMITS", [Limit("query:test", 2, 60)])
    for _ in range(2):
        assert (await client.post("/api/query", json=body(), headers={"X-LLM-Api-Key": KEY})).status_code == 200
    r = await client.post("/api/query", json=body(), headers={"X-LLM-Api-Key": KEY})
    assert r.status_code == 429
    assert int(r.headers["retry-after"]) >= 1


async def test_capacity_sheds_load_with_503(client, monkeypatch):
    monkeypatch.setattr(server.state, "gate", ConcurrencyGate(0, 2))
    r = await client.post("/api/query", json=body(), headers={"X-LLM-Api-Key": KEY})
    assert r.status_code == 503 and r.headers["retry-after"]


async def test_per_client_concurrency_cap(client, monkeypatch):
    gate = ConcurrencyGate(10, 1)
    monkeypatch.setattr(server.state, "gate", gate)
    held, _ = gate.try_acquire("127.0.0.1")
    r = await client.post("/api/query", json=body(), headers={"X-LLM-Api-Key": KEY})
    assert r.status_code == 429
    held.release()


async def test_session_id_cannot_be_shared_workspace(client):
    r = await client.post("/api/query", json={**body(), "session_id": "shared"}, headers={"X-LLM-Api-Key": KEY})
    assert r.status_code == 422


async def test_oversized_message_rejected(client):
    r = await client.post("/api/query", json=body("x" * 5000), headers={"X-LLM-Api-Key": KEY})
    assert r.status_code == 422


async def test_ingest_disabled_on_demo(client):
    r = await client.post(
        "/api/ingest", json={"session_id": SESSION, "value": "2401.15884"}, headers={"X-LLM-Api-Key": KEY}
    )
    assert r.status_code == 403


async def test_corpus_reads_are_cacheable_and_shared_only(client):
    r = await client.get("/api/papers", params={"session_id": SESSION})
    assert r.status_code == 200 and r.json()["count"] == 4
    assert "public" in r.headers["cache-control"]
    g = (await client.get("/api/graph")).json()
    assert {p["id"] for p in g["papers"]} == {p["id"] for p in r.json()["papers"]}
    assert all(not p["own"] for p in g["papers"])


async def test_config_and_health(client):
    cfg = (await client.get("/api/config")).json()
    assert cfg["require_byok"] is True and cfg["ingest_enabled"] is False
    h = (await client.get("/api/health")).json()
    assert h["status"] == "ok" and h["runs_in_flight"] == 0


async def test_health_stays_responsive_under_many_open_streams(client, monkeypatch):
    """v1 parked a thread per open stream and froze at ~30 concurrent
    queries. Here 60 streams are open at once and /api/health must still
    answer quickly."""
    monkeypatch.setattr(server.state, "gate", ConcurrencyGate(200, 200))
    monkeypatch.setattr(server, "QUERY_LIMITS", [Limit("query:load", 10_000, 60)])
    import mock_run

    monkeypatch.setattr(mock_run, "LATENCY_S", 0.5)

    async def one():
        r = await client.post("/api/query", json=body(), headers={"X-LLM-Api-Key": KEY})
        return r.status_code

    tasks = [asyncio.create_task(one()) for _ in range(60)]
    await asyncio.sleep(0.3)
    t0 = time.perf_counter()
    h = await client.get("/api/health")
    health_latency = time.perf_counter() - t0
    statuses = await asyncio.gather(*tasks)
    assert h.status_code == 200 and health_latency < 0.5
    assert statuses == [200] * 60
    assert server.state.gate.in_flight == 0


def test_redaction():
    text = f"401 for url https://x/v1/models?key=AIza{'b' * 30}&pageSize=1 using Bearer {KEY} and {KEY}"
    out = redact(text, KEY)
    assert KEY not in out and "AIza" not in out and "[REDACTED]" in out
