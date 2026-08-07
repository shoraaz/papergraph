"""
etl.py — PaperGraph Ingestion Pipeline
======================================

Pipeline: arXiv search/download -> parse -> section-aware chunk -> embed (Qdrant)
-> LLM concept & citation extraction -> graph upsert (Neo4j).

Features:
  - Section-aware chunking for claim vs result separation.
  - Idempotent upserts for Qdrant & Neo4j.
  - BYOK ModelChoice support for concept extraction.

For full rationale, arXiv search mechanics, and structured-output validation,
see `docs/DESIGN_DECISIONS.md`.

Usage:
    python etl.py "retrieval augmented generation" --max-papers 15
    python etl.py --ids 2406.16167v1 2408.08535v1   # targeted re-ingest
"""

import argparse
import re
import time
from pathlib import Path
from typing import List, Optional

import arxiv
import fitz  # PyMuPDF
import requests
from agno.agent import Agent
from agno.knowledge.chunking.strategy import ChunkingStrategy
from agno.knowledge.document import Document as KnowledgeDocument
from agno.knowledge.reader.text_reader import TextReader
from pydantic import BaseModel, Field

from agents import ModelChoice, _build_model
from db import NEO4J_DATABASE, SHARED_WORKSPACE, get_knowledge, get_neo4j_driver

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PDF_DIR = Path("data/pdfs")
PDF_DIR.mkdir(parents=True, exist_ok=True)

ARXIV_DELAY_SECONDS = 3.0  # per arXiv's API etiquette guidance
ARXIV_NUM_RETRIES = 8  # generous — arXiv API has intermittent 429/503 under load
CHUNK_TARGET_CHARS = 1800  # roughly 400-450 tokens, a reasonable retrieval unit

# Section headings we try to detect. Matched case-insensitively against
# a whole (short, stripped) line. Order doesn't affect matching logic.
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
_COMPILED_SECTION_PATTERNS = [
    (name, re.compile(pattern, re.IGNORECASE)) for name, pattern in SECTION_PATTERNS.items()
]


# ---------------------------------------------------------------------------
# Step 1: Fetch — arXiv search -> metadata + downloaded PDFs
# ---------------------------------------------------------------------------

def search_arxiv(query: str, max_papers: int) -> List[arxiv.Result]:
    """
    Search arXiv by keyword query. Returns metadata only (no PDFs yet).
    Multi-word queries are quoted for exact-phrase matching against the
    abstract field — see module docstring for why this matters.
    """
    quoted_query = query if query.strip().startswith('"') else f'"{query}"'
    client = arxiv.Client(delay_seconds=ARXIV_DELAY_SECONDS, num_retries=ARXIV_NUM_RETRIES)
    search = arxiv.Search(
        query=f"abs:{quoted_query}",
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results = list(client.results(search))
    print(f"[etl] arXiv search abs:{quoted_query} -> {len(results)} papers")
    return results


def fetch_by_ids(ids: List[str]) -> List[arxiv.Result]:
    """Fetch specific papers by arXiv ID — for targeted re-ingestion of
    a paper that failed a previous batch run, without re-running a
    whole search query. Also used by server.py's /ingest for a single
    user-supplied arXiv ID."""
    client = arxiv.Client(delay_seconds=ARXIV_DELAY_SECONDS, num_retries=ARXIV_NUM_RETRIES)
    search = arxiv.Search(id_list=ids)
    results = list(client.results(search))
    print(f"[etl] fetched {len(results)}/{len(ids)} requested papers by ID")
    return results


def download_pdf(paper: arxiv.Result) -> Path:
    """
    Download a paper's PDF to data/pdfs/, skipping if already present.
    Uses paper.pdf_url directly (arxiv==4.0.0 removed the old
    Result.download_pdf() convenience method).
    """
    paper_id = paper.get_short_id()  # e.g. '2005.11401v1'
    pdf_path = PDF_DIR / f"{paper_id}.pdf"
    if pdf_path.exists():
        return pdf_path

    headers = {"User-Agent": "PaperGraph/0.1 (portfolio project; contact via GitHub)"}
    response = requests.get(paper.pdf_url, headers=headers, timeout=30)
    response.raise_for_status()
    pdf_path.write_bytes(response.content)

    time.sleep(ARXIV_DELAY_SECONDS)
    return pdf_path


# ---------------------------------------------------------------------------
# Step 2: Parse — PDF -> plain text, then section-tagged text blocks
# ---------------------------------------------------------------------------

def extract_text(pdf_path: Path) -> str:
    """Extract plain text from a PDF, page by page, joined together."""
    doc = fitz.open(pdf_path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n".join(pages)


def tag_sections(full_text: str) -> List[dict]:
    """
    Walk the extracted text line by line, tracking which section we're
    currently in based on heading matches. Returns a list of
    {"section": str, "text": str} blocks — one per detected section.
    Text before the first recognized heading is tagged "abstract" (the
    common case for arXiv papers, where the abstract precedes any
    numbered heading).
    """
    lines = full_text.split("\n")
    blocks: List[dict] = []
    current_section = "abstract"
    current_lines: List[str] = []

    def flush():
        text = "\n".join(current_lines).strip()
        if text:
            blocks.append({"section": current_section, "text": text})

    for line in lines:
        matched_section = None
        stripped = line.strip()
        if 0 < len(stripped) < 60:  # headings are short lines; skip the regex check on long lines
            for name, pattern in _COMPILED_SECTION_PATTERNS:
                if pattern.match(stripped):
                    matched_section = name
                    break
        if matched_section:
            flush()
            current_section = matched_section
            current_lines = []
        else:
            current_lines.append(line)

    flush()
    return blocks


# ---------------------------------------------------------------------------
# Step 3: Chunk — section-aware, custom ChunkingStrategy for TextReader
# ---------------------------------------------------------------------------

class SectionAwareChunking(ChunkingStrategy):
    """
    Re-detects section boundaries in the document's raw text (via
    tag_sections) and splits into ~CHUNK_TARGET_CHARS pieces without
    letting a chunk span two different sections. Each resulting chunk
    carries a `section` metadata field — this is what lets
    ClaimVerifierAgent search "results section only" rather than the
    whole paper.

    Passed to TextReader(chunking_strategy=...) — Agno's Reader pipeline
    calls .chunk(document) with document.content set to the raw text
    passed into Knowledge.insert(text_content=...).
    """

    def chunk(self, document: KnowledgeDocument) -> List[KnowledgeDocument]:
        section_blocks = tag_sections(document.content or "")
        chunks: List[KnowledgeDocument] = []
        chunk_index = 0

        for block in section_blocks:
            section = block["section"]
            text = block["text"]
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            buffer = ""
            for para in paragraphs:
                if buffer and len(buffer) + len(para) > CHUNK_TARGET_CHARS:
                    chunks.append(self._make_chunk(document, buffer, section, chunk_index))
                    chunk_index += 1
                    buffer = para
                else:
                    buffer = f"{buffer}\n\n{para}" if buffer else para
            if buffer:
                chunks.append(self._make_chunk(document, buffer, section, chunk_index))
                chunk_index += 1

        return chunks

    @staticmethod
    def _make_chunk(document: KnowledgeDocument, text: str, section: str, index: int) -> KnowledgeDocument:
        meta = dict(document.meta_data or {})
        meta["section"] = section
        meta["chunk_index"] = index
        return KnowledgeDocument(
            name=f"{document.name}_chunk_{index}",
            content=text,
            meta_data=meta,
        )


# ---------------------------------------------------------------------------
# Step 4: Graph extraction — LLM call per paper: concepts + relations
# ---------------------------------------------------------------------------

class ConceptExtraction(BaseModel):
    """What a paper teaches vs. assumes, for the concept-dependency graph."""
    teaches: List[str] = Field(
        description="Concepts this paper explains well enough to learn from (3-8 short names, e.g. 'dense retrieval')."
    )
    requires: List[str] = Field(
        description="Concepts this paper assumes the reader already knows (3-8 short names)."
    )


def _extraction_agent(model_choice: Optional[ModelChoice] = None) -> Agent:
    """
    model_choice: which provider/model/key to run concept extraction on.
    None falls back to _build_model's own default (Gemini via
    GOOGLE_API_KEY), same as every other agent in agents.py — but a
    real ModelChoice from a BYOK request should always be passed
    through from ingest_paper() so this respects the user's provider
    selection instead of silently defaulting to Gemini regardless.
    """
    return Agent(
        name="ConceptExtractor",
        model=_build_model(model_choice),
        output_schema=ConceptExtraction,
        instructions=[
            "Given a paper's abstract and introduction, identify:",
            "1. Concepts this paper TEACHES — ideas it explains clearly enough that someone could learn them here.",
            "2. Concepts this paper REQUIRES — background the reader is assumed to already have.",
            "Use short, consistent, lowercase concept names (e.g. 'attention mechanism', not "
            "'the attention mechanism as introduced in transformers').",
            "2-4 items per list. Be extremely selective — extract only 2 to 4 essential core concepts per paper to keep the graph minimal and clean.",
        ],
    )


def extract_concepts(
    abstract: str, intro_text: str, model_choice: Optional[ModelChoice] = None
) -> ConceptExtraction:
    """
    Runs the one real LLM call in the ingestion pipeline. Validates the
    result type immediately rather than trusting it — see the module
    docstring's IMPORTANT note for the real cross-provider failure this
    guards against (some OpenRouter-routed models/providers reject
    structured-output requests outright; Agno's Agent.run() doesn't
    always raise cleanly on that, so result.content can come back as
    something other than a ConceptExtraction instance).
    """
    agent = _extraction_agent(model_choice)
    prompt = f"Abstract:\n{abstract}\n\nIntroduction (may be truncated):\n{intro_text[:3000]}"
    result = agent.run(prompt)

    if not isinstance(result.content, ConceptExtraction):
        model_label = model_choice.resolved_model_id() if model_choice else "default"
        raise ValueError(
            f"Concept extraction did not return a structured ConceptExtraction object "
            f"(got {type(result.content).__name__} instead) using model '{model_label}'. "
            f"This usually means the selected provider/model doesn't support structured "
            f"outputs — try a different model (e.g. a Gemini or well-known OpenAI/Anthropic "
            f"model via OpenRouter) rather than retrying the same one."
        )

    return result.content


# ---------------------------------------------------------------------------
# Step 5: Graph upsert — Neo4j MERGE (idempotent)
# ---------------------------------------------------------------------------

def upsert_paper_graph(
    paper_id: str,
    title: str,
    authors: List[str],
    abstract: str,
    published_date: str,
    arxiv_url: str,
    concepts: ConceptExtraction,
    cited_ids: List[str],
    workspace_id: str = SHARED_WORKSPACE,
) -> None:
    driver = get_neo4j_driver()
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            session.run(
                """
                MERGE (p:Paper {paper_id: $paper_id})
                SET p.title = $title,
                    p.authors = $authors,
                    p.abstract = $abstract,
                    p.published_date = $published_date,
                    p.arxiv_url = $arxiv_url,
                    p.workspace_id = $workspace_id
                """,
                paper_id=paper_id,
                title=title,
                authors=authors,
                abstract=abstract,
                published_date=published_date,
                arxiv_url=arxiv_url,
                workspace_id=workspace_id,
            )

            for concept_name in concepts.teaches:
                session.run(
                    """
                    MERGE (c:Concept {name: $name})
                    ON CREATE SET c.display_name = $display_name
                    WITH c
                    MATCH (p:Paper {paper_id: $paper_id})
                    MERGE (p)-[:TEACHES]->(c)
                    """,
                    name=concept_name.lower().strip(),
                    display_name=concept_name.strip(),
                    paper_id=paper_id,
                )

            for concept_name in concepts.requires:
                session.run(
                    """
                    MERGE (c:Concept {name: $name})
                    ON CREATE SET c.display_name = $display_name
                    WITH c
                    MATCH (p:Paper {paper_id: $paper_id})
                    MERGE (p)-[:REQUIRES]->(c)
                    """,
                    name=concept_name.lower().strip(),
                    display_name=concept_name.strip(),
                    paper_id=paper_id,
                )

            for cited_id in cited_ids:
                # NOTE: a cited paper referenced here but not itself
                # ingested gets created as a bare stub node (no
                # workspace_id set). This is intentional — citation
                # stubs from a user's own upload shouldn't silently
                # claim "shared" or leak into another workspace's view,
                # and Concept nodes/queries never filter on Paper
                # workspace_id for citation stubs anyway since they
                # have no chunks or TEACHES/REQUIRES edges to leak.
                session.run(
                    """
                    MATCH (from:Paper {paper_id: $from_id})
                    MERGE (to:Paper {paper_id: $to_id})
                    MERGE (from)-[:CITES]->(to)
                    """,
                    from_id=paper_id,
                    to_id=cited_id,
                )
    finally:
        driver.close()


def extract_arxiv_citations(text: str, own_id: str) -> List[str]:
    """
    Cheap heuristic: find arXiv IDs (e.g. 2005.11401) mentioned in the
    paper's own text/references, excluding self-references. This is
    intentionally simple — a proper citation graph would use arXiv's
    reference API or Semantic Scholar, a reasonable future upgrade but
    overkill for an initial 15-20 paper corpus where most cited works
    won't be in the corpus anyway.
    """
    ids = set(re.findall(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b", text))
    ids.discard(own_id.split("v")[0])
    return list(ids)[:20]  # cap — a references section can match many false positives


# ---------------------------------------------------------------------------
# Orchestration: one paper, start to finish
# ---------------------------------------------------------------------------

def ingest_paper(
    paper: arxiv.Result,
    workspace_id: str = SHARED_WORKSPACE,
    model_choice: Optional[ModelChoice] = None,
) -> ConceptExtraction:
    """
    Full pipeline for one paper: download, parse, chunk+embed (tagged
    with workspace_id, always local/free — see module docstring),
    extract concepts (LLM call, honors model_choice), upsert into the
    graph (also tagged with workspace_id). Returns the extracted
    concepts so callers (CLI or ingest_paper_for_workspace) can
    report/log without a second lookup.
    """
    paper_id = paper.get_short_id()
    print(f"[etl] ingesting {paper_id} into workspace='{workspace_id}': {paper.title[:70]}...")

    pdf_path = download_pdf(paper)
    full_text = extract_text(pdf_path)
    section_blocks = tag_sections(full_text)

    # Vector store: section-aware chunks, via the public Knowledge.insert
    # API with a custom TextReader(chunking_strategy=SectionAwareChunking()).
    # skip_if_exists relies on Knowledge's own content-hash tracking, so
    # re-running etl.py on the same paper won't duplicate chunks. This
    # step never touches any LLM provider — embeddings are always local
    # (BAAI/bge-base-en-v1.5 via db.py's get_embedder()).
    knowledge = get_knowledge(workspace_id=workspace_id)
    knowledge.insert(
        name=paper_id,
        text_content=full_text,
        metadata={
            "paper_id": paper_id,
            "title": paper.title,
            "arxiv_url": paper.entry_id,
            "workspace_id": workspace_id,
        },
        reader=TextReader(chunking_strategy=SectionAwareChunking()),
        skip_if_exists=True,
    )

    # Graph: concept extraction (the one real LLM call in this pipeline,
    # honors model_choice, validated for type — see extract_concepts())
    # + upsert.
    abstract_block = next((b["text"] for b in section_blocks if b["section"] == "abstract"), paper.summary)
    intro_block = next((b["text"] for b in section_blocks if b["section"] == "introduction"), "")
    concepts = extract_concepts(abstract_block, intro_block, model_choice=model_choice)
    cited_ids = extract_arxiv_citations(full_text, paper_id)

    upsert_paper_graph(
        paper_id=paper_id,
        title=paper.title,
        authors=[a.name for a in paper.authors],
        abstract=paper.summary,
        published_date=str(paper.published.date()) if paper.published else "",
        arxiv_url=paper.entry_id,
        concepts=concepts,
        cited_ids=cited_ids,
        workspace_id=workspace_id,
    )

    print(
        f"[etl] {paper_id}: teaches={concepts.teaches}, requires={concepts.requires}, "
        f"cites={len(cited_ids)} in-corpus-candidate IDs"
    )
    return concepts


def ingest_paper_for_workspace(
    paper: arxiv.Result,
    workspace_id: str,
    model_choice: Optional[ModelChoice] = None,
) -> dict:
    """
    Non-CLI entrypoint for server.py's /ingest. Same underlying pipeline
    as ingest_paper(), but returns a plain result dict (for the SSE
    response) instead of only printing, and requires an explicit
    workspace_id (no SHARED_WORKSPACE default — a live per-session
    upload should never silently land in the shared corpus).

    model_choice: passed straight through to ingest_paper() so the
    concept-extraction LLM call actually honors whichever provider the
    request's BYOK sheet specified, instead of always hitting Gemini
    regardless of the user's selection (see module docstring).
    """
    if not workspace_id or workspace_id == SHARED_WORKSPACE:
        raise ValueError(
            "ingest_paper_for_workspace requires a real session workspace_id, "
            "not empty or the shared workspace — uploads must be session-isolated."
        )

    concepts = ingest_paper(paper, workspace_id=workspace_id, model_choice=model_choice)
    return {
        "paper_id": paper.get_short_id(),
        "title": paper.title,
        "workspace_id": workspace_id,
        "teaches": concepts.teaches,
        "requires": concepts.requires,
    }


def run_etl(query: str, max_papers: int) -> None:
    papers = search_arxiv(query, max_papers)
    _ingest_all(papers)


def run_etl_by_ids(ids: List[str]) -> None:
    """Targeted re-ingest — for filling gaps left by a partial failure
    in a previous batch run, without re-searching or re-touching papers
    that already succeeded. Always uses SHARED_WORKSPACE — this is a
    CLI maintenance operation on the base corpus, not a user upload."""
    papers = fetch_by_ids(ids)
    _ingest_all(papers)


def _ingest_all(papers: List[arxiv.Result], workspace_id: str = SHARED_WORKSPACE) -> None:
    for i, paper in enumerate(papers, 1):
        print(f"\n[etl] --- paper {i}/{len(papers)} ---")
        try:
            ingest_paper(paper, workspace_id=workspace_id)
        except Exception as e:
            print(f"[etl] FAILED on {paper.get_short_id()}: {e}")
            continue


EASY_5_PAPERS = [
    "2309.01431",  # Benchmarking RAG (RGB) - Noise & Rejection
    "2401.15884",  # Corrective RAG (CRAG) - Evaluator & Corrective Search
    "2310.11511",  # Self-RAG - Self-reflection tokens & grading
    "2405.13002",  # DuetRAG - Parametric + Non-parametric Memory
    "2412.15404",  # RAG Literature Navigation Framework
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest arXiv papers into PaperGraph")
    parser.add_argument("query", nargs="?", help="arXiv search query, e.g. 'retrieval augmented generation'")
    parser.add_argument("--max-papers", type=int, default=5)
    parser.add_argument(
        "--ids", nargs="+", help="Specific arXiv IDs to (re-)ingest instead of running a search query"
    )
    args = parser.parse_args()

    if args.ids:
        run_etl_by_ids(args.ids)
    elif args.query:
        run_etl(args.query, args.max_papers)
    else:
        print("[etl] Ingesting 5 curated, easy foundational RAG papers...")
        run_etl_by_ids(EASY_5_PAPERS)
