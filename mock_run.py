"""
mock_run.py — a scripted agent run for load testing (MOCK_LLM=true)
===================================================================

Load-testing the real pipeline would spend real LLM quota and mostly
measure the provider. This mode replaces only the LLM: it emits the same
SSE event shapes as a real run, sleeps where model calls would happen
(MOCK_LATENCY_S per "model call"), and performs REAL retrieval — an
embedded/hosted Qdrant hybrid search and a graph lookup — through the
same typed tools the agents use.

So a load test in this mode exercises everything the server owns:
SSE fan-out, rate limiting, the concurrency gate, query embedding, vector
search and graph reads. Numbers from it describe the serving path, not
LLM throughput, and docs/SYSTEM_DESIGN.md reports them that way.

Never enable in production: it answers without a model.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import AsyncIterator

from event_mapper import COLORS, GLYPHS
from tools import Toolset

LATENCY_S = float(os.environ.get("MOCK_LATENCY_S", "0.6"))


async def mock_events(tools: Toolset, message: str) -> AsyncIterator[tuple[str, dict]]:
    t0 = time.perf_counter()
    yield "step", {
        "kind": "plan", "glyph": GLYPHS["plan"], "color": COLORS["plan"],
        "title": "Team leader delegates to Retriever (mock)", "body": message[:200], "ms": "",
        "detailLabel": "MOCK", "detail": ["MOCK_LLM=true — scripted run, real retrieval"],
    }
    await asyncio.sleep(LATENCY_S)

    t = time.perf_counter()
    found = json.loads(await tools.search_corpus(message))
    yield "step", {
        "kind": "retrieve", "glyph": GLYPHS["retrieve_vector"], "color": COLORS["retrieve_vector"],
        "title": "Vector store · search_corpus · Retriever (mock)",
        "body": f"Retrieved {found['count']} chunk(s).", "ms": f"{time.perf_counter() - t:.2f}s",
        "query": message[:200], "detailLabel": "RETRIEVED CHUNKS",
        "detail": [f"{r['paper_id']} · {r['section']}" for r in found["results"][:6]] or ["(nothing returned)"],
    }

    t = time.perf_counter()
    papers = json.loads(await tools.find_papers(message[:80]))
    yield "step", {
        "kind": "retrieve", "glyph": GLYPHS["retrieve_graph"], "color": COLORS["retrieve_graph"],
        "title": "Knowledge graph · find_papers · Retriever (mock)",
        "body": f"Graph lookup returned {len(papers['papers'])} result(s).", "ms": f"{time.perf_counter() - t:.2f}s",
        "query": message[:80], "detailLabel": "GRAPH RESULT",
        "detail": [p["title"] for p in papers["papers"]] or ["(nothing returned)"],
    }
    await asyncio.sleep(LATENCY_S)

    score = 0.8 if found["count"] else 0.1
    yield "step", {
        "kind": "grade", "glyph": GLYPHS["grade_sufficient"], "color": COLORS["grade_sufficient"],
        "title": "Self-RAG grader · mock", "body": "Scripted grade.", "ms": "", "score": score,
        "verdict": "sufficient" if score >= 0.5 else "insufficient", "detailLabel": "", "detail": [],
    }

    answer = "This is a scripted MOCK_LLM answer. Top passages came from: " + ", ".join(
        sorted({r["paper_id"] for r in found["results"]})
    ) + "."
    for word in answer.split(" "):
        yield "delta", {"text": word + " "}
        await asyncio.sleep(LATENCY_S / 20)
    yield "answer", {"answer": answer, "claims": [], "elapsed_s": round(time.perf_counter() - t0, 2)}
