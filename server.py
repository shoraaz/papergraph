"""
server.py — PaperGraph HTTP/SSE Backend
=========================================

FastAPI application fronting the Agno agent team:
  - POST /query            : Real-time SSE stream of reasoning steps & token deltas.
  - POST /ingest           : Ingest paper into caller's isolated workspace.
  - GET  /models/{provider}: Live model list for BYOK selection.
  - GET  /graph            : Scoped Neo4j graph nodes & layout coordinates.
  - GET  /papers           : Flat paper list for sidebar.

For details on streaming architecture, queue bridging, and model filtering,
see `docs/DESIGN_DECISIONS.md`.
"""

import asyncio
import json
import math
import os
import queue
import threading
from collections import defaultdict
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from agents import ModelChoice, build_papergraph_team
from db import NEO4J_DATABASE, SHARED_WORKSPACE, close_neo4j_driver, get_neo4j_driver
from event_mapper import RunTraceBuilder

app = FastAPI(title="PaperGraph API")

# Wide-open CORS for local dev / a single-page demo frontend. Tighten
# this to a specific origin once the frontend has a fixed deploy URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_SENTINEL = object()  # marks "the worker thread is done" on the queue

# Groq serves non-chat models (speech-to-text, text-to-speech, safety
# classifiers) from the SAME /models endpoint as its chat-completion
# models, with no capability field to distinguish them programmatically.
# Denylist by known family name substring — see the IMPORTANT note above
# for why this matters (a selected non-chat model hangs the whole query
# pipeline silently).
_GROQ_NON_CHAT_MARKERS = ("whisper", "tts", "orpheus", "prompt-guard", "llama-guard", "guard")


def _is_non_chat_groq_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return any(marker in lowered for marker in _GROQ_NON_CHAT_MARKERS)


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class ProviderKey(BaseModel):
    provider: str = "gemini"  # "gemini" | "groq" | "openrouter"
    model_id: Optional[str] = None
    api_key: Optional[str] = None  # BYOK key; None = use server env var (demo/free mode)


class QueryRequest(BaseModel):
    message: str
    session_id: str
    model: ProviderKey = ProviderKey()


class IngestRequest(BaseModel):
    session_id: str
    mode: str  # "arxiv" | "url"
    value: str  # arXiv ID or PDF URL
    model: ProviderKey = ProviderKey()  # used for the concept-extraction LLM call


def _model_choice_from(provider_key: ProviderKey) -> ModelChoice:
    return ModelChoice(
        provider=provider_key.provider,
        model_id=provider_key.model_id,
        api_key=provider_key.api_key,
    )


# ---------------------------------------------------------------------------
# /query — SSE stream of the real reasoning trace, true incremental delivery
# ---------------------------------------------------------------------------

@app.post("/query")
async def query(req: QueryRequest):
    model_choice = _model_choice_from(req.model)

    async def event_stream():
        builder = RunTraceBuilder()
        event_queue: queue.Queue = queue.Queue()

        def _worker():
            """Runs in a background thread. Pushes each raw Agno event onto
            the queue AS IT'S PRODUCED (not after the whole run finishes),
            then a sentinel, or an exception object if the run itself blew
            up before producing any events at all."""
            try:
                team = build_papergraph_team(session_user_id=req.session_id, model_choice=model_choice)
                for event in team.run(req.message, stream=True, stream_events=True):
                    event_queue.put(event)
            except Exception as e:
                event_queue.put(e)
            finally:
                event_queue.put(_SENTINEL)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        try:
            while True:
                item = await asyncio.to_thread(event_queue.get)

                if item is _SENTINEL:
                    break

                if isinstance(item, Exception):
                    yield {
                        "event": "step",
                        "data": json.dumps(
                            {
                                "kind": "error",
                                "glyph": "!",
                                "color": "var(--red)",
                                "title": "Server error",
                                "body": str(item)[:500],
                                "ms": "",
                                "detailLabel": "RAW ERROR",
                                "detail": [str(item)[:2000]],
                            }
                        ),
                    }
                    yield {"event": "done", "data": json.dumps({"errored": True})}
                    return

                steps, deltas = builder.feed(item)

                for delta in deltas:
                    yield {"event": "delta", "data": json.dumps({"text": delta})}

                for step in steps:
                    yield {"event": "step", "data": json.dumps(step.to_dict())}
                    if step.kind == "error":
                        # Stop after the first surfaced provider error rather
                        # than continue draining the queue — avoids relaying
                        # the cascading Agno crash noise that follows a
                        # provider failure (see event_mapper.py's module
                        # docstring for the NoneType.to_dict() bug this
                        # sidesteps). The worker thread keeps running to
                        # completion in the background and is simply
                        # abandoned; it's a daemon thread so it won't block
                        # process exit.
                        yield {"event": "done", "data": json.dumps({"errored": True})}
                        return

            # Clean completion: send the final (complete/authoritative)
            # answer + any claim cards. The frontend already has the
            # streamed text from 'delta' events; this is the fallback/
            # summary version plus data (claims) only available now.
            yield {
                "event": "answer",
                "data": json.dumps({"answer": builder.final_answer, "claims": builder.claims}),
            }
            yield {"event": "done", "data": json.dumps({"errored": False})}

        except Exception as e:
            # Defense in depth: something in the SSE loop itself (not the
            # Agno run) raised. Still give the client a clean error rather
            # than a dropped connection.
            yield {
                "event": "step",
                "data": json.dumps(
                    {
                        "kind": "error",
                        "glyph": "!",
                        "color": "var(--red)",
                        "title": "Server error",
                        "body": str(e)[:500],
                        "ms": "",
                        "detailLabel": "RAW ERROR",
                        "detail": [str(e)[:2000]],
                    }
                ),
            }
            yield {"event": "done", "data": json.dumps({"errored": True})}

    return EventSourceResponse(event_stream())


# ---------------------------------------------------------------------------
# /ingest — per-session paper upload
# ---------------------------------------------------------------------------

@app.post("/ingest")
async def ingest(req: IngestRequest):
    """
    Ingests one paper (by arXiv ID or PDF URL) into the caller's own
    isolated workspace (workspace_id = session_id). Streams the same
    stage-progress shape the UI's mock INGEST_STAGES already expects:
    fetch/parse -> chunk -> embed -> graph-extract.

    model_choice is built from req.model and passed all the way through
    to ingest_paper_for_workspace() -> ingest_paper() -> extract_concepts()
    -> _extraction_agent(), so the concept-extraction LLM call actually
    uses the provider the user selected in the BYOK sheet — see the
    module docstring here and in etl.py for the real bug this fixes
    (ingestion silently always hit Gemini's quota regardless of BYOK
    selection until this was wired through).
    """
    model_choice = _model_choice_from(req.model)

    from etl import fetch_by_ids, ingest_paper_for_workspace  # local import: avoid pulling in etl.py's heavier deps (fitz, arxiv) unless /ingest is actually used

    async def stage_stream():
        try:
            if req.mode == "arxiv":
                papers = fetch_by_ids([req.value])
            elif req.mode == "url":
                raise HTTPException(status_code=400, detail="PDF URL ingest not yet implemented")
            else:
                raise HTTPException(status_code=400, detail=f"Unknown ingest mode: {req.mode}")

            if not papers:
                yield {"event": "error", "data": json.dumps({"message": "Could not fetch that paper."})}
                return

            paper = papers[0]

            def _ingest_sync():
                return ingest_paper_for_workspace(paper, workspace_id=req.session_id, model_choice=model_choice)

            result = await asyncio.to_thread(_ingest_sync)
            yield {"event": "complete", "data": json.dumps(result)}

        except Exception as e:
            yield {"event": "error", "data": json.dumps({"message": str(e)[:500]})}

    return EventSourceResponse(stage_stream())


# ---------------------------------------------------------------------------
# /models/{provider} — live model list for the BYOK chooser
# ---------------------------------------------------------------------------

@app.get("/models/{provider}")
async def list_models(provider: str, api_key: Optional[str] = None):
    """
    OpenRouter's model list is a public, unauthenticated endpoint — can
    be fetched before the user enters any key. Groq and Gemini require
    the caller's own key to list models, matching how BYOK works for
    actually using them: no key, no list, same restriction either way.

    Groq's list is filtered to exclude known non-chat model families
    (speech-to-text, text-to-speech, safety classifiers) — see the
    module docstring's IMPORTANT note for the real bug this prevents.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        if provider == "openrouter":
            resp = await client.get("https://openrouter.ai/api/v1/models")
            resp.raise_for_status()
            data = resp.json()
            models = [
                {"id": m["id"], "name": m.get("name", m["id"])}
                for m in data.get("data", [])
            ]
            return {"models": models}

        if provider == "groq":
            key = api_key or os.environ.get("GROQ_API_KEY")
            if not key:
                return {"models": [], "requires_key": True}
            resp = await client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {key}"},
            )
            resp.raise_for_status()
            data = resp.json()
            models = [
                {"id": m["id"], "name": m["id"]}
                for m in data.get("data", [])
                if not _is_non_chat_groq_model(m["id"])
            ]
            return {"models": models}

        if provider == "gemini":
            key = api_key or os.environ.get("GOOGLE_API_KEY")
            if not key:
                return {"models": [], "requires_key": True}
            resp = await client.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params={"key": key},
            )
            resp.raise_for_status()
            data = resp.json()
            models = [
                {"id": m["name"].removeprefix("models/"), "name": m.get("displayName", m["name"])}
                for m in data.get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])
            ]
            return {"models": models}

        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# /papers — flat sidebar list, workspace-scoped
# ---------------------------------------------------------------------------

@app.get("/papers")
async def list_papers(session_id: str):
    """Every paper visible to this session: the shared base corpus plus
    any of their own uploads. Excludes citation-stub nodes (papers that
    only exist because another paper cites them, never actually
    ingested — no chunks, no TEACHES/REQUIRES edges) since those aren't
    real corpus entries the sidebar should list."""

    def _query():
        for attempt in range(2):
            try:
                driver = get_neo4j_driver()
                with driver.session(database=NEO4J_DATABASE) as session:
                    result = session.run(
                        """
                        MATCH (p:Paper)
                        WHERE p.workspace_id IN [$workspace_id, $shared]
                          AND (p.title IS NOT NULL AND p.title <> '')
                        RETURN p.paper_id AS paper_id, p.title AS title,
                               p.workspace_id AS workspace_id, p.arxiv_url AS arxiv_url
                        ORDER BY p.workspace_id = $shared DESC, p.title
                        """,
                        workspace_id=session_id,
                        shared=SHARED_WORKSPACE,
                    )
                    return [dict(record) for record in result], None
            except Exception as e:
                print(f"[server] Neo4j connection error in /papers (attempt {attempt + 1}): {e}")
                close_neo4j_driver()
                if attempt == 1:
                    return [], str(e)

    rows, err = await asyncio.to_thread(_query)
    if err:
        raise HTTPException(status_code=500, detail=f"Neo4j database connection error: {err}")
    papers = [
        {
            "id": row["paper_id"],
            "title": row["title"],
            "arxiv": (row["paper_id"] or "").split("v")[0],
            "own": row["workspace_id"] != SHARED_WORKSPACE,
        }
        for row in rows
    ]
    return {"papers": papers, "count": len(papers)}


# ---------------------------------------------------------------------------
# /graph — full papers + concepts + edges, workspace-scoped, with layout
# ---------------------------------------------------------------------------

@app.get("/graph")
async def get_graph(session_id: str):
    """
    Real Neo4j data for the graph visualization: every paper visible to
    this session (shared + their own), every concept those papers
    TEACH/REQUIRE, and the edges themselves. Computes simple x/y
    coordinates server-side (see module docstring) so the frontend
    doesn't need its own layout engine.
    """

    def _query():
        for attempt in range(2):
            try:
                driver = get_neo4j_driver()
                with driver.session(database=NEO4J_DATABASE) as session:
                    papers_result = session.run(
                        """
                        MATCH (p:Paper)
                        WHERE p.workspace_id IN [$workspace_id, $shared]
                          AND (p.title IS NOT NULL AND p.title <> '')
                        RETURN p.paper_id AS paper_id, p.title AS title, p.workspace_id AS workspace_id
                        ORDER BY p.title
                        """,
                        workspace_id=session_id,
                        shared=SHARED_WORKSPACE,
                    )
                    papers = [dict(r) for r in papers_result]
                    paper_ids = [p["paper_id"] for p in papers]

                    edges_result = session.run(
                        """
                        MATCH (p:Paper)-[r:TEACHES|REQUIRES]->(c:Concept)
                        WHERE p.paper_id IN $paper_ids
                        RETURN p.paper_id AS paper_id, c.name AS concept_id,
                               c.display_name AS concept_label, type(r) AS rel_type
                        """,
                        paper_ids=paper_ids,
                    )
                    edges = [dict(r) for r in edges_result]

                    return papers, edges, None
            except Exception as e:
                print(f"[server] Neo4j connection error in /graph (attempt {attempt + 1}): {e}")
                close_neo4j_driver()
                if attempt == 1:
                    return [], [], str(e)

    papers, edges, err = await asyncio.to_thread(_query)

    concept_ids = sorted({e["concept_id"] for e in edges})
    concepts = [
        {"id": cid, "label": next(e["concept_label"] for e in edges if e["concept_id"] == cid)}
        for cid in concept_ids
    ]

    # Papers arranged on a wide ellipse in title order
    n_papers = max(len(papers), 1)
    radius_x = 540
    radius_y = 380
    center_x, center_y = 700, 480
    paper_pos = {}
    for i, p in enumerate(papers):
        angle = 2 * math.pi * i / n_papers
        paper_pos[p["paper_id"]] = (
            center_x + radius_x * math.cos(angle),
            center_y + radius_y * math.sin(angle),
        )

    # Place each concept near the centroid of connected papers, then run repulsion iterations
    concept_neighbors = defaultdict(list)
    for e in edges:
        if e["paper_id"] in paper_pos:
            concept_neighbors[e["concept_id"]].append(paper_pos[e["paper_id"]])

    concept_pos = {}
    for cid in concept_ids:
        neighbors = concept_neighbors.get(cid, [])
        if neighbors:
            cx = sum(x for x, y in neighbors) / len(neighbors)
            cy = sum(y for x, y in neighbors) / len(neighbors)
        else:
            cx, cy = center_x, center_y
        concept_pos[cid] = [cx, cy]

    # Iterative repulsion solver to prevent concept card overlapping
    for _ in range(40):
        for i in range(len(concept_ids)):
            c1 = concept_ids[i]
            x1, y1 = concept_pos[c1]
            for j in range(i + 1, len(concept_ids)):
                c2 = concept_ids[j]
                x2, y2 = concept_pos[c2]
                dx = x1 - x2
                dy = y1 - y2
                if abs(dx) < 170 and abs(dy) < 55:
                    push_x = (170 - abs(dx)) * 0.5 * (1.0 if dx >= 0 else -1.0)
                    push_y = (55 - abs(dy)) * 0.5 * (1.0 if dy >= 0 else -1.0)
                    concept_pos[c1][0] += push_x
                    concept_pos[c1][1] += push_y
                    concept_pos[c2][0] -= push_x
                    concept_pos[c2][1] -= push_y

    pos = {**{pid: list(xy) for pid, xy in paper_pos.items()}, **{cid: list(xy) for cid, xy in concept_pos.items()}}

    paper_teaches = defaultdict(list)
    paper_requires = defaultdict(list)
    for e in edges:
        target = paper_teaches if e["rel_type"] == "TEACHES" else paper_requires
        target[e["paper_id"]].append(e["concept_id"])

    return {
        "papers": [
            {
                "id": p["paper_id"],
                "title": p["title"],
                "arxiv": (p["paper_id"] or "").split("v")[0],
                "own": p["workspace_id"] != SHARED_WORKSPACE,
                "teaches": paper_teaches.get(p["paper_id"], []),
                "requires": paper_requires.get(p["paper_id"], []),
            }
            for p in papers
        ],
        "concepts": concepts,
        "pos": pos,
    }


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
