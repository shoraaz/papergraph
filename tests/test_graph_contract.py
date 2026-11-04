"""
One behavioural suite, run against every GraphStore backend.

  snapshot : SnapshotGraphStore built from the fixture corpus
  neo4j    : a real Neo4j (skipped if unreachable), loaded through the
             same upsert_paper() the ETL uses
  neo4j->snapshot : export_snapshot() from that Neo4j, reloaded — proves
             the demo snapshot answers exactly like the source of truth
"""

import os

import pytest

from stores.base import (
    Concept,
    ConceptEdge,
    GraphSnapshot,
    ReadOnlyStoreError,
    build_prerequisite_report,
    normalize_concept,
)
from stores.snapshot_graph import SnapshotGraphStore
from tests.fixtures import OTHER, OWNER, PAPERS, SHARED

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "papergraph")


def fixture_snapshot() -> GraphSnapshot:
    concepts: dict[str, Concept] = {}
    edges: list[ConceptEdge] = []
    cites: list[tuple[str, str]] = []
    for paper, teaches, requires, cited in PAPERS:
        for rel, names in (("TEACHES", teaches), ("REQUIRES", requires)):
            for n in names:
                key = normalize_concept(n)
                concepts.setdefault(key, Concept(key, n))
                edges.append(ConceptEdge(paper.paper_id, rel, key))
        cites += [(paper.paper_id, c) for c in cited]
    return GraphSnapshot([p for p, *_ in PAPERS], list(concepts.values()), edges, cites)


async def _neo4j_store():
    neo4j = pytest.importorskip("neo4j")
    from stores.neo4j_store import Neo4jGraphStore

    store = Neo4jGraphStore(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        await store.ensure_ready()
    except Exception as e:  # pragma: no cover - environment dependent
        await store.close()
        pytest.skip(f"Neo4j not reachable at {NEO4J_URI}: {e}")
    async with store._driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
    # Self-RAG first: its citations create version-less stubs that the
    # later inserts of RAG / DPR must absorb.
    order = sorted(PAPERS, key=lambda row: row[0].paper_id != "2310.11511v1")
    for paper, teaches, requires, cited in order:
        await store.upsert_paper(paper, teaches, requires, cited)
    return store


@pytest.fixture(params=["snapshot", "neo4j", "neo4j->snapshot"])
async def graph(request):
    if request.param == "snapshot":
        yield SnapshotGraphStore(fixture_snapshot())
        return
    store = await _neo4j_store()
    try:
        if request.param == "neo4j":
            yield store
        else:
            snap = await store.export_snapshot(workspaces=[SHARED, OWNER])
            yield SnapshotGraphStore(snap)
    finally:
        await store.close()


async def test_find_papers_by_id_title_and_words(graph):
    by_id = await graph.find_papers("arXiv:2005.11401", [SHARED])
    assert [p.paper_id for p in by_id] == ["2005.11401v4"]
    by_title = await graph.find_papers("dense passage retrieval", [SHARED])
    assert [p.paper_id for p in by_title] == ["2004.04906v3"]
    by_words = await graph.find_papers("critique self-reflection", [SHARED])
    assert [p.paper_id for p in by_words] == ["2310.11511v1"]
    assert await graph.find_papers("nonexistent title words", [SHARED]) == []


async def test_workspace_isolation(graph):
    assert await graph.find_papers("instruction tuning", [SHARED]) == []
    assert await graph.find_papers("instruction tuning", [SHARED, OTHER]) == []
    assert await graph.get_paper("2501.00001", [SHARED, OTHER]) is None
    own = await graph.find_papers("instruction tuning", [SHARED, OWNER])
    assert [p.paper_id for p in own] == ["2501.00001v1"]
    listed = [p.paper_id for p in await graph.list_papers([SHARED])]
    assert "2501.00001v1" not in listed and len(listed) == 4
    teaching = await graph.papers_teaching("instruction tuning", [SHARED, OTHER])
    assert teaching == []


async def test_get_paper_details_and_citations(graph):
    d = await graph.get_paper("2310.11511", [SHARED])
    assert d["title"].startswith("Self-RAG")
    assert d["teaches"] == ["self-reflection tokens"]
    assert set(d["requires"]) == {"retrieval augmented generation", "dense retrieval", "instruction tuning"}
    # Cited by base id; must resolve to the versioned, ingested nodes.
    assert [c["paper_id"] for c in d["cites_in_corpus"]] == ["2004.04906v3", "2005.11401v4"]
    rag = await graph.get_paper("2005.11401v4", [SHARED])
    assert [c["paper_id"] for c in rag["cited_by_in_corpus"]] == ["2310.11511v1"]


async def test_prerequisites_partial_when_teacher_is_private(graph):
    report = build_prerequisite_report(await graph.prerequisite_inputs("2310.11511", [SHARED, OTHER]))
    assert report["paper_found"] is True
    assert report["outcome"] == "partial"
    assert report["untaught_concepts"] == ["instruction tuning"]
    assert [s["paper_id"] for s in report["reading_order"]] == ["2005.11401v4", "2310.11511v1"]


async def test_prerequisites_complete_for_owner_and_ordered(graph):
    report = build_prerequisite_report(await graph.prerequisite_inputs("2310.11511", [SHARED, OWNER]))
    assert report["outcome"] == "complete"
    # The upload REQUIRES rag, which 2005.11401 TEACHES -> RAG must come first.
    assert [s["paper_id"] for s in report["reading_order"]] == ["2005.11401v4", "2501.00001v1", "2310.11511v1"]


async def test_prerequisites_not_found(graph):
    report = build_prerequisite_report(await graph.prerequisite_inputs("1234.56789", [SHARED]))
    assert report == {
        "paper_found": False,
        "message": "No paper matching that id/title exists in this workspace's corpus.",
    }


async def test_papers_teaching_fuzzy(graph):
    rows = await graph.papers_teaching("Dense Retrieval", [SHARED])
    assert sorted(r["paper_id"] for r in rows) == ["2004.04906v3", "2005.11401v4"]


async def test_snapshot_is_read_only():
    store = SnapshotGraphStore(fixture_snapshot())
    with pytest.raises(ReadOnlyStoreError):
        await store.upsert_paper(PAPERS[0][0], [], [], [])
