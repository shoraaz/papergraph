"""
graph_view.py — papers + concepts + layout for the graph tab
============================================================

Shared by GET /api/graph (live, includes a session's own uploads) and by
`python etl.py export`, which writes the same payload for the shared
corpus to frontend/data/corpus.json. The frontend reads that static file
first, so the sidebar and graph render straight from the CDN, before the
API process has even woken up.

Layout (unchanged from v1): papers on an ellipse in title order, each
concept at the centroid of the papers it touches, then a few rounds of
pairwise repulsion so concept cards don't overlap.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from config import SHARED_WORKSPACE
from stores.base import GraphStore, base_arxiv_id


async def build_graph_view(graph: GraphStore, workspaces: Sequence[str]) -> dict:
    papers = await graph.list_papers(workspaces)
    papers = sorted(papers, key=lambda p: p.title)
    edges = await graph.concept_edges_for([p.paper_id for p in papers])

    concept_label: dict[str, str] = {}
    teaches: dict[str, list[str]] = defaultdict(list)
    requires: dict[str, list[str]] = defaultdict(list)
    for edge, concept in edges:
        concept_label.setdefault(concept.name, concept.display_name)
        (teaches if edge.rel == "TEACHES" else requires)[edge.paper_id].append(concept.name)

    concept_ids = sorted(concept_label)
    pos = _layout([p.paper_id for p in papers], concept_ids, teaches, requires)

    return {
        "papers": [
            {
                "id": p.paper_id,
                "title": p.title,
                "arxiv": base_arxiv_id(p.paper_id),
                "own": p.workspace_id != SHARED_WORKSPACE,
                "teaches": teaches.get(p.paper_id, []),
                "requires": requires.get(p.paper_id, []),
            }
            for p in papers
        ],
        "concepts": [{"id": cid, "label": concept_label[cid]} for cid in concept_ids],
        "pos": pos,
    }


def papers_payload(view: dict) -> dict:
    """The sidebar list, derived from a graph view (one source of truth)."""
    papers = sorted(view["papers"], key=lambda p: (p["own"], p["title"]))  # shared corpus first
    papers = [{k: p[k] for k in ("id", "title", "arxiv", "own")} for p in papers]
    return {"papers": papers, "count": len(papers)}


def _layout(
    paper_ids: list[str],
    concept_ids: list[str],
    teaches: dict[str, list[str]],
    requires: dict[str, list[str]],
) -> dict:
    n = max(len(paper_ids), 1)
    rx, ry, cx0, cy0 = 540, 380, 700, 480
    paper_pos = {
        pid: (cx0 + rx * math.cos(2 * math.pi * i / n), cy0 + ry * math.sin(2 * math.pi * i / n))
        for i, pid in enumerate(paper_ids)
    }

    neighbors: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for pid in paper_ids:
        for cid in teaches.get(pid, []) + requires.get(pid, []):
            neighbors[cid].append(paper_pos[pid])

    concept_pos: dict[str, list[float]] = {}
    for cid in concept_ids:
        pts = neighbors.get(cid, [])
        if pts:
            concept_pos[cid] = [sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts)]
        else:
            concept_pos[cid] = [cx0, cy0]

    for _ in range(40):
        for i, c1 in enumerate(concept_ids):
            for c2 in concept_ids[i + 1 :]:
                x1, y1 = concept_pos[c1]
                x2, y2 = concept_pos[c2]
                dx, dy = x1 - x2, y1 - y2
                if abs(dx) < 170 and abs(dy) < 55:
                    px = (170 - abs(dx)) * 0.5 * (1.0 if dx >= 0 else -1.0)
                    py = (55 - abs(dy)) * 0.5 * (1.0 if dy >= 0 else -1.0)
                    concept_pos[c1][0] += px
                    concept_pos[c1][1] += py
                    concept_pos[c2][0] -= px
                    concept_pos[c2][1] -= py

    out = {pid: [round(x, 1), round(y, 1)] for pid, (x, y) in paper_pos.items()}
    out.update({cid: [round(x, 1), round(y, 1)] for cid, (x, y) in concept_pos.items()})
    return out
