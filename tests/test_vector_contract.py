"""
One behavioural suite for QdrantVectorStore, run against:

  embedded : AsyncQdrantClient(":memory:") — the demo profile
  server   : a real Qdrant server (skipped if unreachable) — the full profile
  snapshot : export from the server, import into embedded — what the demo
             actually serves after `etl.py export`
"""

import os
import uuid

import pytest
from qdrant_client import AsyncQdrantClient

from embeddings import HashingEmbedder
from stores.qdrant_store import QdrantVectorStore
from tests.fixtures import CHUNKS, OTHER, OWNER, SHARED

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


async def _server_store() -> QdrantVectorStore:
    client = AsyncQdrantClient(url=QDRANT_URL, timeout=5)
    try:
        await client.get_collections()
    except Exception as e:  # pragma: no cover - environment dependent
        await client.close()
        pytest.skip(f"Qdrant not reachable at {QDRANT_URL}: {e}")
    store = QdrantVectorStore(client, f"test_{uuid.uuid4().hex[:8]}", HashingEmbedder(384), embedded=False)
    await store.ensure_ready()
    await store.upsert_chunks(CHUNKS)
    return store


@pytest.fixture(params=["embedded", "server", "snapshot"])
async def vstore(request):
    if request.param == "embedded":
        store = QdrantVectorStore(AsyncQdrantClient(location=":memory:"), "t", HashingEmbedder(384), embedded=True)
        await store.ensure_ready()
        await store.upsert_chunks(CHUNKS)
        yield store
        await store.close()
        return
    server = await _server_store()
    try:
        if request.param == "server":
            yield server
        else:
            points = await server.export_points(workspaces=[SHARED, OWNER])
            store = QdrantVectorStore(AsyncQdrantClient(location=":memory:"), "t", HashingEmbedder(384), embedded=True)
            await store.ensure_ready()
            assert await store.import_points(points) == len(CHUNKS)
            yield store
            await store.close()
    finally:
        await server.client.delete_collection(server.collection)
        await server.close()


async def test_hybrid_search_ranks_matching_chunk_first(vstore):
    hits = await vstore.search("self-reflection tokens retrieve critique", [SHARED], k=3)
    assert hits and hits[0].paper_id == "2310.11511v1"


async def test_filters_paper_and_section(vstore):
    hits = await vstore.search("benchmarks results", [SHARED], k=5, paper_id="2310.11511", sections=["results"])
    assert hits and all(h.paper_id == "2310.11511v1" and h.section == "results" for h in hits)


async def test_upload_is_invisible_to_other_workspaces(vstore):
    for ws in ([SHARED], [SHARED, OTHER]):
        hits = await vstore.search("zebra quokka secret upload", ws, k=10)
        assert all(h.paper_id != "2501.00001v1" for h in hits)
    own = await vstore.search("zebra quokka secret upload", [SHARED, OWNER], k=3)
    assert own[0].paper_id == "2501.00001v1"


async def test_unscoped_search_is_refused(vstore):
    with pytest.raises(ValueError):
        await vstore.search("anything", [], k=3)


async def test_reingest_is_idempotent(vstore):
    before = await vstore.count()
    await vstore.upsert_chunks(CHUNKS[:2])
    assert await vstore.count() == before


async def test_export_defaults_to_shared_only():
    store = QdrantVectorStore(AsyncQdrantClient(location=":memory:"), "t", HashingEmbedder(384), embedded=True)
    await store.ensure_ready()
    await store.upsert_chunks(CHUNKS)
    exported = await store.export_points()
    assert exported and all(p["payload"]["workspace_id"] == SHARED for p in exported)
    await store.close()
