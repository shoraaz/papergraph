"""
etl.py — PaperGraph ingestion + snapshot export
===============================================

    arXiv fetch -> PDF parse -> section-aware chunks -> Qdrant (dense + BM25)
                             -> LLM concept extraction -> Neo4j (TEACHES / REQUIRES / CITES)

and, for the zero-cost public demo:

    Neo4j + Qdrant  --export-->  snapshot/   (graph.json, chunks.jsonl.gz, manifest.json)
                                 frontend/data/corpus.json  (sidebar + graph, served by the CDN)

Neo4j and Qdrant remain the system of record: ingestion only ever writes
there (STORE_PROFILE=full). The snapshot is a derived, read-only artifact
— regenerate it whenever the corpus changes and commit it.

Usage (with `docker compose up -d qdrant neo4j` running, or hosted URLs in .env):

    STORE_PROFILE=full REQUIRE_BYOK=false GOOGLE_API_KEY=... python etl.py ingest            # curated corpus
    STORE_PROFILE=full REQUIRE_BYOK=false GOOGLE_API_KEY=... python etl.py ingest --ids 2310.11511
    STORE_PROFILE=full REQUIRE_BYOK=false GOOGLE_API_KEY=... python etl.py ingest --query "corrective rag" --max-papers 5
    STORE_PROFILE=full python etl.py export                                                  # -> snapshot/ + frontend/data/

Ingestion-only dependencies (arxiv, pymupdf, requests) live in
requirements-etl.txt so the serving bundle stays small.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import re
import time
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from config import ROOT, SHARED_WORKSPACE, settings
from stores import CHUNKS_FILE, GRAPH_FILE, MANIFEST, Stores, open_stores
from stores.base import ChunkRecord, Paper, normalize_paper_ref

PDF_DIR = ROOT / "data" / "pdfs"
ARXIV_DELAY_SECONDS = 3.0  # arXiv API etiquette
ARXIV_NUM_RETRIES = 8
CHUNK_TARGET_CHARS = 1800  # ~400-450 tokens
# References are excluded from the vector index: they are dense lists of
# titles and arXiv ids that match many queries and answer none of them —
# a likely contributor to v1's 0.42 context precision. (Citation ids are
# still extracted from the full text for the graph.)
INDEX_SKIP_SECTIONS = {"references"}

SECTION_PATTERNS = {
    "abstract": r"^\s*abstract\s*$",
    "introduction": r"^\s*(\d+\.?\s*)?introduction\s*$",
    "related_work": r"^\s*(\d+\.?\s*)?related\s+work\s*$",
    "methods": r"^\s*(\d+\.?\s*)?(method(ology|s)?|approach)\s*$",
    "results": r"^\s*(\d+\.?\s*)?(results|experiments?(\s+and\s+results)?)\s*$",
    "discussion": r"^\s*(\d+\.?\s*)?discussion\s*$",
    "conclusion": r"^\s*(\d+\.?\s*)?conclusion(s)?\s*$",
    "references": r"^\s*(\d+\.?\s*)?references\s*$",
}
_COMPILED_SECTION_PATTERNS = [(n, re.compile(p, re.IGNORECASE)) for n, p in SECTION_PATTERNS.items()]

# The v1 corpus: every paper id that appears in the retrieved contexts of
# eval/results.json (the last RAGAS run against the hosted Qdrant/Neo4j),
# so `etl.py ingest` with no arguments rebuilds the same corpus the
# README's numbers were measured on.
CURATED_CORPUS = [
    "2309.01431", "2310.11511", "2401.15884", "2402.11626", "2403.14197",
    "2405.13002", "2405.19893", "2406.03790", "2408.08535", "2409.01864",
    "2411.00437", "2412.15404", "2501.11929", "2502.01113", "2509.10467",
    "2512.17194", "2603.04647", "2604.18509", "2607.01852",
]


class IngestError(RuntimeError):
    """A user-facing ingestion failure (bad id, fetch failed, extraction failed)."""


# ---------------------------------------------------------------------------
# Parse + chunk (pure functions — unit-tested in tests/test_etl.py)
# ---------------------------------------------------------------------------

def tag_sections(full_text: str) -> List[dict]:
    """Split text into {"section", "text"} blocks on recognised headings.
    Text before the first heading is tagged "abstract"."""
    blocks: List[dict] = []
    current, lines = "abstract", []

    def flush() -> None:
        text = "\n".join(lines).strip()
        if text:
            blocks.append({"section": current, "text": text})

    for line in full_text.split("\n"):
        stripped = line.strip()
        matched = None
        if 0 < len(stripped) < 60:
            for name, pattern in _COMPILED_SECTION_PATTERNS:
                if pattern.match(stripped):
                    matched = name
                    break
        if matched:
            flush()
            current, lines = matched, []
        else:
            lines.append(line)
    flush()
    return blocks


def chunk_sections(blocks: List[dict], target_chars: int = CHUNK_TARGET_CHARS) -> List[tuple[str, str]]:
    """(section, text) chunks of ~target_chars that never span two sections."""
    out: List[tuple[str, str]] = []
    for block in blocks:
        if block["section"] in INDEX_SKIP_SECTIONS:
            continue
        paragraphs = [p.strip() for p in block["text"].split("\n\n") if p.strip()]
        buf = ""
        for para in paragraphs:
            if buf and len(buf) + len(para) > target_chars:
                out.append((block["section"], buf))
                buf = para
            else:
                buf = f"{buf}\n\n{para}" if buf else para
        if buf:
            out.append((block["section"], buf))
    return out


def extract_arxiv_citations(text: str, own_id: str) -> List[str]:
    """arXiv ids mentioned in the text (cheap heuristic), minus self."""
    ids = sorted(set(re.findall(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b", text)))
    own = normalize_paper_ref(own_id)
    return [i for i in ids if i != own][:20]


# ---------------------------------------------------------------------------
# Fetch (arXiv) — blocking I/O, run in a thread
# ---------------------------------------------------------------------------

def _fetch_arxiv(ids: List[str]):
    import arxiv

    client = arxiv.Client(delay_seconds=ARXIV_DELAY_SECONDS, num_retries=ARXIV_NUM_RETRIES)
    return list(client.results(arxiv.Search(id_list=ids)))


def _search_arxiv(query: str, max_papers: int):
    import arxiv

    quoted = query if query.strip().startswith('"') else f'"{query}"'
    client = arxiv.Client(delay_seconds=ARXIV_DELAY_SECONDS, num_retries=ARXIV_NUM_RETRIES)
    return list(client.results(arxiv.Search(query=f"abs:{quoted}", max_results=max_papers)))


def _download_and_parse(paper) -> str:
    import fitz  # PyMuPDF
    import requests

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = PDF_DIR / f"{paper.get_short_id()}.pdf"
    if not pdf_path.exists():
        resp = requests.get(
            paper.pdf_url, headers={"User-Agent": "PaperGraph/2.0 (portfolio project)"}, timeout=30
        )
        resp.raise_for_status()
        if len(resp.content) > 40 * 1024 * 1024:
            raise IngestError("PDF is larger than 40 MB.")
        pdf_path.write_bytes(resp.content)
        time.sleep(ARXIV_DELAY_SECONDS)
    doc = fitz.open(pdf_path)
    try:
        return "\n".join(page.get_text("text") for page in doc)
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Concept extraction — the one LLM call in the pipeline
# ---------------------------------------------------------------------------

class ConceptExtraction(BaseModel):
    teaches: List[str] = Field(description="2-4 concepts this paper explains well enough to learn from.")
    requires: List[str] = Field(description="2-4 concepts this paper assumes the reader already knows.")


async def extract_concepts(abstract: str, intro: str, model_choice) -> ConceptExtraction:
    from agno.agent import Agent

    from agents import _build_model

    agent = Agent(
        name="ConceptExtractor",
        model=_build_model(model_choice),
        output_schema=ConceptExtraction,
        instructions=[
            "Given a paper's abstract and introduction, identify:",
            "1. Concepts this paper TEACHES — ideas it explains clearly enough that someone could learn them here.",
            "2. Concepts this paper REQUIRES — background the reader is assumed to already have.",
            "Use short, consistent, lowercase concept names (e.g. 'attention mechanism').",
            "2-4 items per list. Be extremely selective — only the essential core concepts.",
        ],
    )
    result = await agent.arun(f"Abstract:\n{abstract}\n\nIntroduction (may be truncated):\n{intro[:3000]}")
    if not isinstance(result.content, ConceptExtraction):
        raise IngestError(
            f"Concept extraction did not return structured output (got {type(result.content).__name__}) with "
            f"model '{model_choice.resolved_model_id()}'. Try a model that supports structured outputs."
        )
    return result.content


# ---------------------------------------------------------------------------
# Ingest one paper through the stores
# ---------------------------------------------------------------------------

async def ingest_paper(paper, workspace_id: str, model_choice, stores: Stores) -> dict:
    paper_id = paper.get_short_id()
    print(f"[etl] ingesting {paper_id} into workspace='{workspace_id}': {paper.title[:70]}")
    full_text = await asyncio.to_thread(_download_and_parse, paper)
    blocks = tag_sections(full_text)

    chunks = [
        ChunkRecord(paper_id, paper.title, section, i, text, workspace_id, paper.entry_id)
        for i, (section, text) in enumerate(chunk_sections(blocks))
    ]
    n_chunks = await stores.vector.upsert_chunks(chunks)

    abstract = next((b["text"] for b in blocks if b["section"] == "abstract"), paper.summary)
    intro = next((b["text"] for b in blocks if b["section"] == "introduction"), "")
    concepts = await extract_concepts(abstract, intro, model_choice)
    cites = extract_arxiv_citations(full_text, paper_id)

    await stores.graph.upsert_paper(
        Paper(
            paper_id=paper_id,
            title=paper.title,
            workspace_id=workspace_id,
            authors=[a.name for a in paper.authors],
            abstract=paper.summary,
            published_date=str(paper.published.date()) if paper.published else "",
            arxiv_url=paper.entry_id,
        ),
        concepts.teaches,
        concepts.requires,
        cites,
    )
    print(f"[etl] {paper_id}: {n_chunks} chunks, teaches={concepts.teaches}, requires={concepts.requires}")
    return {
        "paper_id": paper_id,
        "title": paper.title,
        "workspace_id": workspace_id,
        "chunks": n_chunks,
        "teaches": concepts.teaches,
        "requires": concepts.requires,
    }


async def ingest_arxiv_id(arxiv_ref: str, workspace_id: str, model_choice, stores: Stores) -> dict:
    """Entry point for POST /api/ingest — a visitor's upload, never into
    the shared corpus."""
    if not workspace_id or workspace_id == SHARED_WORKSPACE:
        raise IngestError("Uploads must go to a session workspace, not the shared corpus.")
    ref = normalize_paper_ref(arxiv_ref)
    if not re.fullmatch(r"\d{4}\.\d{4,5}", ref):
        raise IngestError("That doesn't look like an arXiv id (e.g. 2401.15884).")
    papers = await asyncio.to_thread(_fetch_arxiv, [ref])
    if not papers:
        raise IngestError(f"arXiv has no paper {ref}.")
    return await ingest_paper(papers[0], workspace_id, model_choice, stores)


# ---------------------------------------------------------------------------
# Snapshot export (full profile -> demo artifacts)
# ---------------------------------------------------------------------------

async def export_snapshot(stores: Stores, out_dir: Path, frontend_data: Path) -> dict:
    from embeddings import get_embedder
    from graph_view import build_graph_view

    out_dir.mkdir(parents=True, exist_ok=True)
    snap = await stores.graph.export_snapshot(workspaces=[SHARED_WORKSPACE])
    (out_dir / GRAPH_FILE).write_text(json.dumps(snap.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")

    points = await stores.vector.export_points(workspaces=[SHARED_WORKSPACE])
    with gzip.open(out_dir / CHUNKS_FILE, "wt", encoding="utf-8") as fh:
        for p in sorted(points, key=lambda p: p["id"]):
            fh.write(json.dumps(p, ensure_ascii=False, separators=(",", ":")) + "\n")

    embedder = get_embedder()
    manifest = {
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dense_model": embedder.name,
        "dense_dim": embedder.dim,
        "sparse_model": settings.sparse_model,
        "papers": len(snap.papers),
        "concepts": len(snap.concepts),
        "chunks": len(points),
        "source": {"qdrant": settings.qdrant_url, "neo4j": settings.neo4j_uri.split("@")[-1]},
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    view = await build_graph_view(stores.graph, [SHARED_WORKSPACE])
    frontend_data.mkdir(parents=True, exist_ok=True)
    (frontend_data / "corpus.json").write_text(
        json.dumps({"exported_at": manifest["exported_at"], **view}, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[etl] exported {manifest['papers']} papers, {manifest['chunks']} chunks -> {out_dir}")
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def _cli(args: argparse.Namespace) -> None:
    from agents import ModelChoice

    if settings.store_profile != "full":
        raise SystemExit("Run the ETL with STORE_PROFILE=full (it writes to Qdrant + Neo4j).")
    stores = await open_stores()
    try:
        if args.command == "export":
            await export_snapshot(stores, Path(args.out), Path(args.frontend_data))
            return
        choice = ModelChoice(provider=args.provider, model_id=args.model)
        if args.query:
            papers = await asyncio.to_thread(_search_arxiv, args.query, args.max_papers)
        else:
            papers = await asyncio.to_thread(_fetch_arxiv, args.ids or CURATED_CORPUS)
        print(f"[etl] {len(papers)} paper(s) to ingest")
        for i, paper in enumerate(papers, 1):
            print(f"\n[etl] --- paper {i}/{len(papers)} ---")
            try:
                await ingest_paper(paper, SHARED_WORKSPACE, choice, stores)
            except Exception as e:  # keep going; re-run with --ids to fill gaps
                print(f"[etl] FAILED on {paper.get_short_id()}: {e}")
    finally:
        await stores.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="PaperGraph ETL")
    sub = parser.add_subparsers(dest="command", required=True)
    ing = sub.add_parser("ingest", help="ingest papers into the shared corpus (Qdrant + Neo4j)")
    ing.add_argument("--ids", nargs="+", help="arXiv ids (default: the curated corpus)")
    ing.add_argument("--query", help="arXiv abstract search instead of ids")
    ing.add_argument("--max-papers", type=int, default=5)
    ing.add_argument("--provider", default="gemini", choices=["gemini", "groq", "openrouter"])
    ing.add_argument("--model", default=None, help="model id for concept extraction")
    exp = sub.add_parser("export", help="export the shared corpus to the demo snapshot")
    exp.add_argument("--out", default=str(settings.snapshot_dir))
    exp.add_argument("--frontend-data", default=str(ROOT / "frontend" / "data"))
    asyncio.run(_cli(parser.parse_args()))


if __name__ == "__main__":
    main()
