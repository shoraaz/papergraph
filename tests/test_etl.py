"""ETL: parsing/chunking, and the export -> demo-load round trip."""

import dataclasses
import json

from qdrant_client import AsyncQdrantClient

from config import settings
from embeddings import HashingEmbedder
from etl import chunk_sections, export_snapshot, extract_arxiv_citations, tag_sections
from stores import Stores, open_stores
from stores.qdrant_store import QdrantVectorStore
from stores.snapshot_graph import SnapshotGraphStore
from tests.fixtures import CHUNKS, SHARED
from tests.test_graph_contract import fixture_snapshot

PAPER_TEXT = """Self-RAG: a title line
We introduce a framework.

1 Introduction
Large language models hallucinate.

4 Experiments and Results
Self-RAG outperforms baselines on six tasks.

References
[1] Lewis et al. 2005.11401
[2] Karpukhin et al. arXiv:2004.04906v3
"""


def test_tag_sections_and_chunking_skip_references():
    blocks = tag_sections(PAPER_TEXT)
    assert [b["section"] for b in blocks] == ["abstract", "introduction", "results", "references"]
    chunks = chunk_sections(blocks)
    assert [s for s, _ in chunks] == ["abstract", "introduction", "results"]


def test_chunks_never_span_sections_and_respect_size():
    text = "Abstract\n" + "\n\n".join(["alpha " * 60] * 10) + "\nResults\n" + "beta " * 30
    chunks = chunk_sections(tag_sections(text), target_chars=800)
    assert all(len(t) <= 800 + 400 for _, t in chunks)
    assert {s for s, _ in chunks} == {"abstract", "results"}
    assert all("beta" not in t for s, t in chunks if s == "abstract")


def test_citations_exclude_self_and_dedupe():
    assert extract_arxiv_citations(PAPER_TEXT + " 2310.11511v2", "2310.11511v1") == ["2004.04906", "2005.11401"]


async def test_export_then_demo_profile_serves_same_answers(tmp_path, monkeypatch):
    # Source stores (stand-ins for Qdrant server + Neo4j; the contract tests
    # prove those behave identically).
    v = QdrantVectorStore(AsyncQdrantClient(location=":memory:"), "src", HashingEmbedder(384), embedded=True)
    await v.ensure_ready()
    await v.upsert_chunks(CHUNKS)
    src = Stores(v, SnapshotGraphStore(fixture_snapshot()), "full", {})

    manifest = await export_snapshot(src, tmp_path / "snapshot", tmp_path / "frontend_data")
    assert manifest["chunks"] == sum(1 for c in CHUNKS if c.workspace_id == SHARED)
    corpus = json.loads((tmp_path / "frontend_data" / "corpus.json").read_text())
    assert len(corpus["papers"]) == 4 and corpus["pos"]
    graph_json = json.loads((tmp_path / "snapshot" / "graph.json").read_text())
    assert all(p["workspace_id"] == SHARED for p in graph_json["papers"])

    cfg = dataclasses.replace(settings, store_profile="demo", snapshot_dir=tmp_path / "snapshot")
    demo = await open_stores(cfg)
    try:
        assert demo.info["chunks"] == manifest["chunks"] and demo.info["papers"] == 4
        hits = await demo.vector.search("self-reflection tokens critique", [SHARED], k=2)
        assert hits[0].paper_id == "2310.11511v1"
        assert [p.paper_id for p in await demo.graph.find_papers("dense passage", [SHARED])] == ["2004.04906v3"]
    finally:
        await demo.close()
        await src.close()
