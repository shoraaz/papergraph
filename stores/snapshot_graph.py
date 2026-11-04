"""
stores/snapshot_graph.py — the knowledge graph, served from memory
=================================================================

The demo profile's GraphStore. Loaded from snapshot/graph.json, which is
exported *from Neo4j* by `python etl.py export` — Neo4j stays the
system of record, this is a read replica that costs nothing to host.

Why a snapshot instead of a hosted Neo4j for the public demo:
  * the corpus graph is small (tens of papers, ~100 concepts) and only
    changes when the ETL runs — it is effectively static data;
  * a free AuraDB instance pauses after a few idle days and is deleted
    after a few weeks paused, which is exactly what took the demo down;
  * every API instance holds its own copy, so graph reads add no
    network hop, no connection pool and no shared bottleneck under load.

It answers the same typed questions as Neo4jGraphStore through the same
`build_prerequisite_report`, and tests/test_graph_contract.py runs one
suite against both to keep them equivalent.

Writes raise ReadOnlyStoreError: uploads need the full profile.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

from stores.base import (
    Concept,
    ConceptEdge,
    GraphSnapshot,
    Paper,
    PrereqInputs,
    ReadOnlyStoreError,
    base_arxiv_id,
    looks_like_arxiv_id,
    normalize_concept,
    normalize_paper_ref,
    title_matches,
)


class SnapshotGraphStore:
    read_only = True

    def __init__(self, snapshot: GraphSnapshot):
        self._load(snapshot)

    @classmethod
    def from_file(cls, path: Path) -> "SnapshotGraphStore":
        if not path.exists():
            print(f"[graph] snapshot {path} not found — starting with an empty graph")
            return cls(GraphSnapshot(papers=[], concepts=[], concept_edges=[], cites=[]))
        return cls(GraphSnapshot.from_json(json.loads(path.read_text(encoding="utf-8"))))

    def _load(self, snap: GraphSnapshot) -> None:
        # Only papers with a title are real corpus entries; untitled nodes
        # are citation stubs (cited, never ingested).
        self.papers: dict[str, Paper] = {p.paper_id: p for p in snap.papers if p.title}
        self.concepts: dict[str, Concept] = {c.name: c for c in snap.concepts}
        self.teaches: dict[str, set[str]] = defaultdict(set)  # paper -> concepts
        self.requires: dict[str, set[str]] = defaultdict(set)
        self.taught_by: dict[str, set[str]] = defaultdict(set)  # concept -> papers
        for e in snap.concept_edges:
            if e.paper_id not in self.papers:
                continue
            if e.rel == "TEACHES":
                self.teaches[e.paper_id].add(e.concept)
                self.taught_by[e.concept].add(e.paper_id)
            elif e.rel == "REQUIRES":
                self.requires[e.paper_id].add(e.concept)
        # CITES targets may be version-less stubs ("2005.11401") while the
        # ingested node is "2005.11401v4" — resolve by base id so in-corpus
        # citations actually connect.
        by_base = {base_arxiv_id(pid): pid for pid in self.papers}
        self.cites: dict[str, set[str]] = defaultdict(set)
        self.cited_by: dict[str, set[str]] = defaultdict(set)
        for a, b in snap.cites:
            b = b if b in self.papers else by_base.get(base_arxiv_id(b), b)
            if a == b:
                continue
            self.cites[a].add(b)
            self.cited_by[b].add(a)

    async def ensure_ready(self) -> None:
        return None

    async def close(self) -> None:
        return None

    # -- helpers -------------------------------------------------------------

    def _visible(self, workspaces: Sequence[str]) -> list[Paper]:
        ws = set(workspaces)
        return [p for p in self.papers.values() if p.workspace_id in ws]

    def _resolve(self, paper_ref: str, workspaces: Sequence[str]) -> Optional[Paper]:
        ref = normalize_paper_ref(paper_ref)
        if not ref:
            return None
        matches = [p for p in self._visible(workspaces) if p.paper_id.startswith(ref)]
        if not matches:
            return None
        return max(matches, key=lambda p: p.paper_id)  # latest version

    def _concept(self, name: str) -> Concept:
        return self.concepts.get(name) or Concept(name=name, display_name=name)

    # -- reads ---------------------------------------------------------------

    async def find_papers(self, text: str, workspaces: Sequence[str], limit: int = 5) -> list[Paper]:
        visible = self._visible(workspaces)
        if looks_like_arxiv_id(text):
            ref = normalize_paper_ref(text)
            hits = [p for p in visible if p.paper_id.startswith(ref)]
        else:
            hits = [p for p in visible if title_matches(p.title, text)]
        return sorted(hits, key=lambda p: p.title)[:limit]

    async def get_paper(self, paper_ref: str, workspaces: Sequence[str]) -> Optional[dict]:
        p = self._resolve(paper_ref, workspaces)
        if p is None:
            return None
        visible_ids = {v.paper_id for v in self._visible(workspaces)}
        return {
            **p.summary(),
            "authors": p.authors,
            "abstract": p.abstract,
            "teaches": sorted(self._concept(c).display_name for c in self.teaches.get(p.paper_id, ())),
            "requires": sorted(self._concept(c).display_name for c in self.requires.get(p.paper_id, ())),
            "cites_in_corpus": [
                {"paper_id": c, "title": self.papers[c].title}
                for c in sorted(self.cites.get(p.paper_id, ()))
                if c in visible_ids
            ],
            "cited_by_in_corpus": [
                {"paper_id": c, "title": self.papers[c].title}
                for c in sorted(self.cited_by.get(p.paper_id, ()))
                if c in visible_ids
            ],
        }

    async def prerequisite_inputs(self, paper_ref: str, workspaces: Sequence[str]) -> PrereqInputs:
        target = self._resolve(paper_ref, workspaces)
        if target is None:
            return PrereqInputs(paper=None, required=[], teachers={}, teacher_edges=[])
        visible_ids = {p.paper_id for p in self._visible(workspaces)}
        required = sorted(self.requires.get(target.paper_id, ()))
        teachers: dict[str, list[Paper]] = {}
        teacher_ids: set[str] = set()
        for c in required:
            ps = sorted(
                (self.papers[pid] for pid in self.taught_by.get(c, ()) if pid in visible_ids and pid != target.paper_id),
                key=lambda p: p.paper_id,
            )
            teachers[c] = ps
            teacher_ids |= {p.paper_id for p in ps}
        edges = [ConceptEdge(pid, "TEACHES", c) for pid in sorted(teacher_ids) for c in sorted(self.teaches[pid])]
        edges += [ConceptEdge(pid, "REQUIRES", c) for pid in sorted(teacher_ids) for c in sorted(self.requires[pid])]
        return PrereqInputs(
            paper=target,
            required=[self._concept(c) for c in required],
            teachers=teachers,
            teacher_edges=edges,
        )

    async def papers_teaching(self, concept: str, workspaces: Sequence[str], limit: int = 8) -> list[dict]:
        needle = normalize_concept(concept)
        if not needle:
            return []
        visible_ids = {p.paper_id for p in self._visible(workspaces)}
        out = []
        for name in sorted(self.taught_by):
            if needle in name or name in needle:
                for pid in sorted(self.taught_by[name]):
                    if pid in visible_ids:
                        out.append(
                            {
                                "concept": self._concept(name).display_name,
                                "paper_id": pid,
                                "title": self.papers[pid].title,
                            }
                        )
        return out[:limit]

    async def list_papers(self, workspaces: Sequence[str]) -> list[Paper]:
        from config import SHARED_WORKSPACE

        return sorted(self._visible(workspaces), key=lambda p: (p.workspace_id != SHARED_WORKSPACE, p.title))

    async def concept_edges_for(self, paper_ids: Sequence[str]) -> list[tuple[ConceptEdge, Concept]]:
        out = []
        for pid in paper_ids:
            for c in sorted(self.teaches.get(pid, ())):
                out.append((ConceptEdge(pid, "TEACHES", c), self._concept(c)))
            for c in sorted(self.requires.get(pid, ())):
                out.append((ConceptEdge(pid, "REQUIRES", c), self._concept(c)))
        return out

    # -- writes (not supported) ----------------------------------------------

    async def upsert_paper(self, paper, teaches, requires, cites) -> None:
        raise ReadOnlyStoreError("The demo graph is a read-only snapshot; run with STORE_PROFILE=full to ingest.")

    async def export_snapshot(self, workspaces: Sequence[str] = ()) -> GraphSnapshot:
        edges = [ConceptEdge(p, "TEACHES", c) for p, cs in self.teaches.items() for c in cs]
        edges += [ConceptEdge(p, "REQUIRES", c) for p, cs in self.requires.items() for c in cs]
        return GraphSnapshot(
            papers=list(self.papers.values()),
            concepts=list(self.concepts.values()),
            concept_edges=edges,
            cites=[(a, b) for a, bs in self.cites.items() for b in bs],
        )
