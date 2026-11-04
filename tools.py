"""
tools.py — the only things agents can do to the data
====================================================

Five typed, read-only tools, built per request by `build_toolset()` as
closures over (stores, workspaces). The workspace scope is captured from
the server-validated session id — none of the tool signatures has a
parameter through which the LLM could name a different workspace.

  search_corpus(query, paper_id?, section?)   Qdrant hybrid search
  find_papers(query)                          graph: id / title lookup
  get_paper_details(paper_id)                 graph: concepts + citations
  get_prerequisites(paper_id)                 graph: reading-order plan
  papers_teaching_concept(concept)            graph: who teaches X

Compared with v1 (KnowledgeTools + raw Neo4jTools Cypher):
  * no LLM-authored queries -> no injection, no unbounded traversals,
    no writes from a chat box;
  * outputs are small, structured JSON with explicit flags
    (`paper_found`, `outcome`), so "not in corpus" is a field the model
    reads, not an inference it can get wrong — the README's
    "two models read the same Cypher result differently" bug;
  * bounded: k <= 8 chunks, chunk text truncated, so one tool call can't
    blow up the context window (or the visitor's token bill).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from stores import Stores
from stores.base import SECTIONS, build_prerequisite_report, normalize_paper_ref

MAX_K = 8

# Tool names, shared with event_mapper.py so the trace UI can classify calls.
VECTOR_TOOLS = ("search_corpus",)
GRAPH_TOOLS = ("find_papers", "get_paper_details", "get_prerequisites", "papers_teaching_concept")


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


@dataclass
class Toolset:
    search_corpus: Callable
    find_papers: Callable
    get_paper_details: Callable
    get_prerequisites: Callable
    papers_teaching_concept: Callable

    @property
    def graph(self) -> list[Callable]:
        return [self.find_papers, self.get_paper_details, self.get_prerequisites, self.papers_teaching_concept]


def build_toolset(stores: Stores, workspaces: Sequence[str]) -> Toolset:
    scope = tuple(workspaces)
    if not scope:
        raise ValueError("A toolset must be scoped to at least one workspace")

    async def search_corpus(query: str, paper_id: Optional[str] = None, section: Optional[str] = None) -> str:
        """Hybrid semantic + keyword (BM25) search over the ingested papers' text chunks.
        Use for content questions: what a paper says, how a method works, claims, results, jargon.

        Args:
            query: Natural-language description of the passage you need.
            paper_id: Optional arXiv id (e.g. "2405.13002") to search within one paper only.
            section: Optional section filter, one of: abstract, introduction, related_work, methods, results, discussion, conclusion. Use "results" when verifying claims.
        """
        sections = None
        if section:
            s = section.strip().lower().replace(" ", "_")
            if s not in SECTIONS:
                return _dump({"error": f"unknown section '{section}'", "valid_sections": list(SECTIONS)})
            sections = [s]
        hits = await stores.vector.search(
            query,
            scope,
            k=6,
            paper_id=normalize_paper_ref(paper_id) if paper_id else None,
            sections=sections,
        )
        out = {"query": query, "results": [h.to_dict() for h in hits[:MAX_K]], "count": len(hits)}
        if (paper_id or sections) and len(hits) < 6:
            # Filtered to a small slice of one paper and got fewer hits than
            # asked for: that IS the whole slice. Say so, or models keep
            # rephrasing the query and burn calls on identical results.
            out["note"] = (
                f"These {len(hits)} chunk(s) are everything indexed for this filter. "
                "Re-searching with the same paper/section filter will return the same chunks; "
                "work with what you have."
            )
        return _dump(out)

    async def find_papers(query: str) -> str:
        """Look up papers in the corpus by arXiv id or by title words.
        Always call this before concluding a paper is or isn't in the corpus.

        Args:
            query: An arXiv id ("2310.11511", "arXiv:2310.11511v1") or words from the title.
        """
        papers = await stores.graph.find_papers(query, scope, limit=5)
        return _dump(
            {
                "query": query,
                "found": bool(papers),
                "papers": [p.summary() for p in papers],
            }
        )

    async def get_paper_details(paper_id: str) -> str:
        """Graph facts for one paper: title, authors, abstract, concepts it TEACHES and REQUIRES,
        and which in-corpus papers it cites / is cited by.

        Args:
            paper_id: arXiv id, with or without version suffix.
        """
        details = await stores.graph.get_paper(paper_id, scope)
        if details is None:
            return _dump({"paper_found": False, "paper_id": paper_id})
        abstract = details.get("abstract") or ""
        details["abstract"] = abstract[:1500] + (" …" if len(abstract) > 1500 else "")
        return _dump({"paper_found": True, **details})

    async def get_prerequisites(paper_id: str) -> str:
        """Prerequisite reading plan for a paper from the concept graph: the concepts it REQUIRES,
        which in-corpus papers TEACH each one, and a dependency-ordered reading list.
        `outcome` is "complete", "partial" (some prerequisites have no in-corpus teacher —
        listed in untaught_concepts) or "no_prerequisites_recorded". `paper_found=false`
        means the paper is not in the corpus.

        Args:
            paper_id: arXiv id of the target paper, with or without version suffix.
        """
        inputs = await stores.graph.prerequisite_inputs(paper_id, scope)
        return _dump(build_prerequisite_report(inputs))

    async def papers_teaching_concept(concept: str) -> str:
        """Find in-corpus papers that TEACH a concept (fuzzy, case-insensitive name match).

        Args:
            concept: Concept name, e.g. "dense retrieval".
        """
        rows = await stores.graph.papers_teaching(concept, scope, limit=8)
        return _dump({"concept": concept, "found": bool(rows), "papers": rows})

    return Toolset(
        search_corpus=search_corpus,
        find_papers=find_papers,
        get_paper_details=get_paper_details,
        get_prerequisites=get_prerequisites,
        papers_teaching_concept=papers_teaching_concept,
    )
