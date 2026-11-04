"""The LLM-facing tools: scoping, input handling, output shape."""

import json

import pytest
from qdrant_client import AsyncQdrantClient

from embeddings import HashingEmbedder
from stores import Stores
from stores.qdrant_store import QdrantVectorStore
from stores.snapshot_graph import SnapshotGraphStore
from tests.fixtures import CHUNKS, OTHER, OWNER, SHARED
from tests.test_graph_contract import fixture_snapshot
from tools import GRAPH_TOOLS, VECTOR_TOOLS, build_toolset


@pytest.fixture
async def stores():
    v = QdrantVectorStore(AsyncQdrantClient(location=":memory:"), "t", HashingEmbedder(384), embedded=True)
    await v.ensure_ready()
    await v.upsert_chunks(CHUNKS)
    s = Stores(v, SnapshotGraphStore(fixture_snapshot()), "demo", {})
    yield s
    await s.close()


def test_trace_ui_tool_names_match_toolset():
    # event_mapper.py classifies trace steps by these exact names.
    from tools import Toolset

    assert set(Toolset.__dataclass_fields__) == set(VECTOR_TOOLS + GRAPH_TOOLS)


async def test_every_tool_param_is_workspace_free(stores):
    import inspect

    ts = build_toolset(stores, [SHARED])
    for fn in [ts.search_corpus, *ts.graph]:
        params = inspect.signature(fn).parameters
        assert not any("workspace" in p for p in params), fn.__name__


async def test_other_session_cannot_reach_upload(stores):
    ts = build_toolset(stores, [SHARED, OTHER])
    hits = json.loads(await ts.search_corpus("zebra quokka secret upload"))["results"]
    assert all(h["paper_id"] != "2501.00001v1" for h in hits)
    assert json.loads(await ts.find_papers("2501.00001"))["found"] is False
    assert json.loads(await ts.get_paper_details("2501.00001"))["paper_found"] is False
    assert json.loads(await ts.get_prerequisites("2501.00001"))["paper_found"] is False


async def test_owner_sees_upload(stores):
    ts = build_toolset(stores, [SHARED, OWNER])
    hits = json.loads(await ts.search_corpus("zebra quokka secret upload"))["results"]
    assert hits[0]["paper_id"] == "2501.00001v1"


async def test_section_filter_and_validation(stores):
    ts = build_toolset(stores, [SHARED])
    out = json.loads(await ts.search_corpus("benchmarks", paper_id="arXiv:2310.11511v1", section="Results"))
    assert out["results"] and {r["section"] for r in out["results"]} == {"results"}
    bad = json.loads(await ts.search_corpus("x", section="appendix"))
    assert "error" in bad and "results" in bad["valid_sections"]


async def test_empty_scope_refused(stores):
    with pytest.raises(ValueError):
        build_toolset(stores, [])
