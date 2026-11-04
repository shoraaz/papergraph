"""
stores/base.py — storage contracts + backend-independent graph logic
====================================================================

Two narrow interfaces sit between the agents and any database:

  VectorStore : hybrid (dense + BM25) chunk search, workspace-scoped
  GraphStore  : typed, read-only graph questions, workspace-scoped

Agents never see a database client and never write a query. They call
typed tools (tools.py) that call these interfaces, and every method
takes the caller's `workspaces` explicitly. Isolation is a function
argument the server fills in, not a sentence in a prompt the LLM may or
may not follow (which is how Neo4j isolation worked before, via
LLM-written Cypher).

Implementations:
  stores/qdrant_store.py   — Qdrant, server OR embedded (same class)
  stores/neo4j_store.py    — Neo4j, parameterized Cypher, read txns
  stores/snapshot_graph.py — in-memory graph exported from Neo4j

The reading-order algorithm lives here, once, so Neo4j and the
snapshot produce identical answers for identical data. Each graph
backend only has to fetch the inputs.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional, Protocol, Sequence

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

SECTIONS = (
    "abstract",
    "introduction",
    "related_work",
    "methods",
    "results",
    "discussion",
    "conclusion",
)


@dataclass
class Paper:
    paper_id: str  # with arXiv version suffix, e.g. "2405.13002v1"
    title: str
    workspace_id: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    published_date: str = ""
    arxiv_url: str = ""

    @property
    def base_id(self) -> str:
        return base_arxiv_id(self.paper_id)

    def summary(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "published_date": self.published_date,
            "arxiv_url": self.arxiv_url,
        }

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Concept:
    name: str  # normalized, lowercase — the identity
    display_name: str


@dataclass
class ConceptEdge:
    paper_id: str
    rel: str  # "TEACHES" | "REQUIRES"
    concept: str  # Concept.name


@dataclass
class ChunkRecord:
    """A chunk going INTO the vector store."""

    paper_id: str
    title: str
    section: str
    chunk_index: int
    text: str
    workspace_id: str
    arxiv_url: str = ""


@dataclass
class ChunkHit:
    """A chunk coming OUT of a search."""

    paper_id: str
    title: str
    section: str
    text: str
    score: float

    def to_dict(self, max_chars: int = 1200) -> dict:
        text = self.text if len(self.text) <= max_chars else self.text[:max_chars] + " …"
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "section": self.section,
            "score": round(self.score, 4),
            "text": text,
        }


@dataclass
class GraphSnapshot:
    """Everything needed to rebuild the graph in memory. Exported from
    Neo4j by `python etl.py export`, loaded by SnapshotGraphStore."""

    papers: list[Paper]
    concepts: list[Concept]
    concept_edges: list[ConceptEdge]
    cites: list[tuple[str, str]]  # (from_paper_id, to_paper_id)

    def to_json(self) -> dict:
        return {
            "version": 1,
            "papers": [p.to_dict() for p in self.papers],
            "concepts": [asdict(c) for c in self.concepts],
            "concept_edges": [asdict(e) for e in self.concept_edges],
            "cites": [list(c) for c in self.cites],
        }

    @classmethod
    def from_json(cls, data: dict) -> "GraphSnapshot":
        return cls(
            papers=[Paper(**p) for p in data.get("papers", [])],
            concepts=[Concept(**c) for c in data.get("concepts", [])],
            concept_edges=[ConceptEdge(**e) for e in data.get("concept_edges", [])],
            cites=[(a, b) for a, b in data.get("cites", [])],
        )


@dataclass
class PrereqInputs:
    """Raw facts a backend fetches; `build_prerequisite_report` turns them
    into an answer. Keeps the algorithm out of Cypher and out of Python
    dict-walking, so both backends share it."""

    paper: Optional[Paper]
    required: list[Concept]
    teachers: dict[str, list[Paper]]  # concept name -> papers teaching it (target excluded)
    teacher_edges: list[ConceptEdge]  # TEACHES/REQUIRES edges of every teacher paper


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------

class VectorStore(Protocol):
    async def ensure_ready(self) -> None: ...
    async def search(
        self,
        query: str,
        workspaces: Sequence[str],
        k: int = 6,
        paper_id: Optional[str] = None,
        sections: Optional[Sequence[str]] = None,
    ) -> list[ChunkHit]: ...
    async def upsert_chunks(self, chunks: Sequence[ChunkRecord]) -> int: ...
    async def count(self) -> int: ...
    async def export_points(self, workspaces: Sequence[str] = ...) -> list[dict]: ...
    async def import_points(self, points: Iterable[dict]) -> int: ...
    async def close(self) -> None: ...


class GraphStore(Protocol):
    read_only: bool

    async def ensure_ready(self) -> None: ...
    async def find_papers(self, text: str, workspaces: Sequence[str], limit: int = 5) -> list[Paper]: ...
    async def get_paper(self, paper_ref: str, workspaces: Sequence[str]) -> Optional[dict]: ...
    async def prerequisite_inputs(self, paper_ref: str, workspaces: Sequence[str]) -> PrereqInputs: ...
    async def papers_teaching(self, concept: str, workspaces: Sequence[str], limit: int = 8) -> list[dict]: ...
    async def list_papers(self, workspaces: Sequence[str]) -> list[Paper]: ...
    async def concept_edges_for(self, paper_ids: Sequence[str]) -> list[tuple[ConceptEdge, Concept]]: ...
    async def upsert_paper(
        self, paper: Paper, teaches: Sequence[str], requires: Sequence[str], cites: Sequence[str]
    ) -> None: ...
    async def export_snapshot(self, workspaces: Sequence[str] = ...) -> GraphSnapshot: ...
    async def close(self) -> None: ...


class ReadOnlyStoreError(RuntimeError):
    """Raised when a write is attempted against the demo snapshot."""


# ---------------------------------------------------------------------------
# Helpers shared by every backend
# ---------------------------------------------------------------------------

_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)


def base_arxiv_id(paper_id: str) -> str:
    """'2405.13002v2' -> '2405.13002'. Non-arXiv ids are returned as-is."""
    m = _ARXIV_ID_RE.fullmatch(paper_id or "")
    return m.group(1) if m else (paper_id or "")


def normalize_paper_ref(ref: str) -> str:
    """Accepts '2405.13002', '2405.13002v1', 'arXiv:2405.13002',
    'https://arxiv.org/abs/2405.13002v1' and returns the bare base id
    (no version) for prefix matching — paper_ids carry a version suffix
    and users almost never type it. Non-arXiv refs pass through trimmed."""
    ref = (ref or "").strip()
    m = _ARXIV_ID_RE.search(ref)
    return m.group(1) if m else ref


def looks_like_arxiv_id(text: str) -> bool:
    return bool(_ARXIV_ID_RE.search(text or ""))


def normalize_concept(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def title_matches(title: str, text: str) -> bool:
    """Case-insensitive: whole phrase contained, or every word (>2 chars)
    contained. Mirrors the Cypher used by Neo4jGraphStore.find_papers."""
    t = (title or "").lower()
    q = (text or "").strip().lower()
    if not q:
        return False
    if q in t:
        return True
    words = [w for w in re.findall(r"[a-z0-9\-]+", q) if len(w) > 2]
    return bool(words) and all(w in t for w in words)


def title_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9\-]+", (text or "").lower()) if len(w) > 2]


def build_prerequisite_report(inputs: PrereqInputs) -> dict:
    """
    Turns raw graph facts into a reading plan:

      1. concepts C  = what the target paper REQUIRES
      2. teachers    = in-corpus papers that TEACH each c in C
      3. pick a small set of teachers covering C (greedy set cover:
         the paper teaching the most still-uncovered concepts first,
         ties broken by earliest publication, then id — deterministic)
      4. order the picked papers so a paper comes after any picked paper
         that TEACHES something it REQUIRES (Kahn topological sort,
         cycles broken by publication date)
      5. the target paper goes last

    The three outcomes the agents must keep distinct are explicit flags,
    not something an LLM has to infer from an empty list:
      paper_found=False                 -> hard refusal (not in corpus)
      paper_found, untaught concepts    -> honest partial answer
      paper_found, everything covered   -> full reading order
    """
    if inputs.paper is None:
        return {
            "paper_found": False,
            "message": "No paper matching that id/title exists in this workspace's corpus.",
        }

    target = inputs.paper
    required_names = [c.name for c in inputs.required]
    display = {c.name: c.display_name for c in inputs.required}

    teaches_by_paper: dict[str, set[str]] = {}
    requires_by_paper: dict[str, set[str]] = {}
    for e in inputs.teacher_edges:
        bucket = teaches_by_paper if e.rel == "TEACHES" else requires_by_paper
        bucket.setdefault(e.paper_id, set()).add(e.concept)

    papers_by_id: dict[str, Paper] = {}
    for plist in inputs.teachers.values():
        for p in plist:
            papers_by_id[p.paper_id] = p

    covered: set[str] = set()
    picked: list[str] = []
    teachable = {c for c in required_names if inputs.teachers.get(c)}
    while teachable - covered:
        remaining = teachable - covered

        def gain(pid: str) -> tuple:
            p = papers_by_id[pid]
            return (
                -len(teaches_by_paper.get(pid, set()) & remaining),
                p.published_date or "9999",
                pid,
            )

        candidates = [pid for pid in papers_by_id if pid not in picked]
        candidates = [pid for pid in candidates if teaches_by_paper.get(pid, set()) & remaining]
        if not candidates:
            break
        best = min(candidates, key=gain)
        picked.append(best)
        covered |= teaches_by_paper.get(best, set()) & remaining

    # Topological order among picked papers.
    deps: dict[str, set[str]] = {pid: set() for pid in picked}
    for a in picked:
        for b in picked:
            if a != b and teaches_by_paper.get(a, set()) & requires_by_paper.get(b, set()):
                deps[b].add(a)  # a must come before b
    order: list[str] = []
    pending = set(picked)
    while pending:
        ready = [pid for pid in pending if not (deps[pid] & pending)]
        if not ready:  # cycle — fall back to publication order
            ready = [min(pending, key=lambda pid: (papers_by_id[pid].published_date or "9999", pid))]
        ready.sort(key=lambda pid: (papers_by_id[pid].published_date or "9999", pid))
        nxt = ready[0]
        order.append(nxt)
        pending.remove(nxt)

    reading_order = []
    for i, pid in enumerate(order, 1):
        p = papers_by_id[pid]
        teaches = sorted(teaches_by_paper.get(pid, set()) & set(required_names))
        reading_order.append(
            {
                "step": i,
                "paper_id": p.paper_id,
                "title": p.title,
                "covers": [display.get(c, c) for c in teaches],
            }
        )
    reading_order.append(
        {
            "step": len(order) + 1,
            "paper_id": target.paper_id,
            "title": target.title,
            "covers": ["(target paper)"],
        }
    )

    untaught = [display.get(c, c) for c in required_names if c not in covered]
    return {
        "paper_found": True,
        "paper": target.summary(),
        "prerequisite_concepts": [
            {
                "concept": display.get(c, c),
                "taught_by": [
                    {"paper_id": p.paper_id, "title": p.title} for p in inputs.teachers.get(c, [])
                ],
            }
            for c in required_names
        ],
        "reading_order": reading_order if order else [],
        "untaught_concepts": untaught,
        "outcome": (
            "no_prerequisites_recorded"
            if not required_names
            else "partial" if untaught else "complete"
        ),
    }


def json_safe(obj: Any) -> Any:
    """Neo4j temporal types etc. -> plain strings."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)
