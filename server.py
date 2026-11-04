"""
server.py — PaperGraph HTTP/SSE API
===================================

  POST /api/query            SSE stream: reasoning steps + answer tokens
  POST /api/ingest           SSE stream: add a paper to the caller's workspace (full profile)
  POST /api/models/{provider} model list for the BYOK picker (key in header)
  GET  /api/papers           sidebar list (shared corpus + caller's uploads)
  GET  /api/graph            graph tab payload with layout
  GET  /api/config           feature flags + limits for the frontend
  GET  /api/health           liveness / readiness

Request path for /api/query, cheapest check first so abuse is rejected
before it costs anything:

  1. schema + size limits (pydantic)          -> 422
  2. rate limit per client (sliding window)   -> 429 + Retry-After
  3. visitor key present (BYOK)               -> 401
  4. key accepted by provider (cached check)  -> 401
  5. concurrency slot (global + per client)   -> 503 / 429 + Retry-After
  6. stream the run, bounded by RUN_TIMEOUT_S and tool-call limits;
     a client disconnect cancels the run (stops spending the visitor's
     tokens) and the slot is always released.

v1 ran every query in its own thread and bridged events through
`asyncio.to_thread(queue.get)`, which parked one default-executor thread
per open stream: around 30 concurrent questions exhausted the pool and
froze the whole server, including /health. v2 consumes Agno's native
async stream (`team.arun`), so an open stream costs a coroutine, not a
thread, and runs abandoned by the client are cancelled instead of
orphaned.

Everything is stateless per request (no server-side sessions, no Mem0),
so instances scale horizontally behind any load balancer.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from sse_starlette.sse import EventSourceResponse
from starlette.background import BackgroundTask

from agents import PROVIDERS, MissingKeyError, ModelChoice, Turn, build_papergraph_team
from config import ROOT, SHARED_WORKSPACE, settings
from embeddings import get_embedder, run_in_pool
from event_mapper import COLORS, GLYPHS, RunTraceBuilder
from graph_view import build_graph_view, papers_payload
from ratelimit import ConcurrencyGate, Decision, Limit, MemoryLimiter, RateLimiter, UpstashLimiter
from security import (
    KeyChecker,
    client_id,
    extract_key,
    redact,
    valid_session_id,
    workspaces_for,
)
from stores import Stores, open_stores
from tools import build_toolset

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

QUERY_LIMITS = [
    Limit("query:min", settings.query_limit_per_min, 60),
    Limit("query:hour", settings.query_limit_per_hour, 3600),
]
INGEST_LIMITS = [Limit("ingest:hour", settings.ingest_limit_per_hour, 3600)]
MODELS_LIMITS = [Limit("models:min", settings.models_limit_per_min, 60)]
READ_LIMITS = [Limit("read:min", settings.read_limit_per_min, 60)]

MAX_BODY_BYTES = 64 * 1024

_GROQ_NON_CHAT_MARKERS = ("whisper", "tts", "orpheus", "prompt-guard", "llama-guard", "guard")


# ---------------------------------------------------------------------------
# Process state (initialised once, lazily — serverless runtimes don't
# always deliver ASGI lifespan events, so the first request can do it too)
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self) -> None:
        self.stores: Optional[Stores] = None
        self.shared_view: Optional[dict] = None
        backend = (
            UpstashLimiter(settings.upstash_url, settings.upstash_token)
            if settings.upstash_url and settings.upstash_token
            else MemoryLimiter()
        )
        self.rate_backend = "upstash" if isinstance(backend, UpstashLimiter) else "memory"
        self.limiter = RateLimiter(backend)
        self.gate = ConcurrencyGate(settings.max_concurrent_runs, settings.max_runs_per_client)
        self.keys = KeyChecker(ttl_s=settings.key_check_ttl_s)
        self.http = httpx.AsyncClient(timeout=10.0)
        self.started_at = time.time()
        self._lock = asyncio.Lock()
        self._models_cache: dict[str, tuple[float, list]] = {}

    async def ready(self) -> Stores:
        if self.stores is not None:
            return self.stores
        async with self._lock:
            if self.stores is None:
                stores = await open_stores()
                self.shared_view = await build_graph_view(stores.graph, [SHARED_WORKSPACE])
                self.stores = stores
                # Load the embedding model in the background so the first
                # question doesn't pay for it (frontend pings /api/health on
                # page load to trigger this while the visitor reads).
                asyncio.get_running_loop().create_task(_warm_embedder())
        return self.stores


async def _warm_embedder() -> None:
    try:
        await run_in_pool(get_embedder().warm)
        print("[server] embedder warm")
    except Exception as e:  # pragma: no cover - network dependent
        print(f"[server] embedder warm-up failed (will retry lazily): {e}")


state = AppState()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await state.ready()
    except Exception as e:
        # Start anyway; /api/health reports not-ready and requests retry init.
        print(f"[server] startup init failed: {e}")
    yield
    if state.stores:
        await state.stores.close()
    await state.keys.close()
    await state.http.aclose()


app = FastAPI(title="PaperGraph API", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-LLM-Api-Key"],
    expose_headers=["Retry-After", "X-RateLimit-Limit", "X-RateLimit-Remaining", "X-Request-Id"],
    max_age=600,
)


@app.middleware("http")
async def guard_and_log(request: Request, call_next):
    """Reject oversized bodies early; log one line per request — method,
    path, status, latency. Never headers (the key lives in one), never
    bodies, never query strings."""
    rid = uuid.uuid4().hex[:12]
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_BODY_BYTES:
        return JSONResponse({"detail": "Request body too large."}, status_code=413)
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-Id"] = rid
    if request.url.path.startswith("/api/"):
        print(
            f"[http] {rid} {request.method} {request.url.path} {response.status_code} "
            f"{(time.perf_counter() - t0) * 1000:.0f}ms"
        )
    return response


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ModelSpec(BaseModel):
    provider: Literal["gemini", "groq", "openrouter"] = "gemini"
    model_id: Optional[str] = Field(default=None, max_length=120, pattern=r"^[A-Za-z0-9._:/\-]+$")


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=settings.max_history_chars)


class QueryRequest(BaseModel):
    message: str = Field(min_length=1, max_length=settings.max_message_chars)
    session_id: str
    model: ModelSpec = ModelSpec()
    history: list[HistoryTurn] = Field(default_factory=list, max_length=20)

    @field_validator("session_id")
    @classmethod
    def _session(cls, v: str) -> str:
        if not valid_session_id(v):
            raise ValueError("session_id must be 8-64 chars of [A-Za-z0-9_-]")
        return v


class IngestRequest(BaseModel):
    session_id: str
    mode: Literal["arxiv"] = "arxiv"
    value: str = Field(min_length=4, max_length=200)
    model: ModelSpec = ModelSpec()

    @field_validator("session_id")
    @classmethod
    def _session(cls, v: str) -> str:
        if not valid_session_id(v):
            raise ValueError("session_id must be 8-64 chars of [A-Za-z0-9_-]")
        return v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def enforce_rate(request: Request, limits: list[Limit]) -> Decision:
    decision = await state.limiter.check(client_id(request), limits)
    if not decision.allowed:
        window = "minute" if decision.limit.window_s <= 60 else "hour"
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit reached ({decision.limit.limit} per {window}). Try again in {decision.retry_after_s}s.",
            headers=decision.headers(),
        )
    return decision


def _error_step(title: str, body: str, *secrets: Optional[str]) -> dict:
    body = redact(body, *secrets)
    return {
        "kind": "error",
        "glyph": GLYPHS["error"],
        "color": COLORS["error"],
        "title": title,
        "body": body[:500],
        "ms": "",
        "detailLabel": "RAW ERROR",
        "detail": [body[:2000]],
    }


def _sse(event: str, data: dict) -> dict:
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}


def _truncate_history(history: list[HistoryTurn]) -> list[Turn]:
    turns = history[-settings.max_history_turns :]
    return [Turn(t.role, t.content[: settings.max_history_chars]) for t in turns]


class RunRecorder:
    """RECORD_RUNS_DIR=frontend/replays: saves successful runs in the
    replay format the frontend plays back without a key. Never records
    the key, the session id or the client address."""

    def __init__(self, message: str, model: ModelChoice):
        self.enabled = bool(settings.record_runs_dir)
        self.t0 = time.perf_counter()
        self.message = message
        self.model = model
        self.events: list[dict] = []

    def add(self, event: str, data: dict) -> None:
        if self.enabled:
            self.events.append({"t": int((time.perf_counter() - self.t0) * 1000), "event": event, "data": data})

    def save(self) -> None:
        if not self.enabled or not self.events:
            return
        out_dir = Path(settings.record_runs_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", self.message.lower()).strip("-")[:48] or "run"
        path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}.json"
        path.write_text(
            json.dumps(
                {
                    "question": self.message,
                    "provider": self.model.provider,
                    "model": self.model.resolved_model_id(),
                    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "events": self.events,
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"[server] recorded run -> {path}")


# ---------------------------------------------------------------------------
# /api/query
# ---------------------------------------------------------------------------

@app.post("/api/query")
async def query(req: QueryRequest, request: Request):
    stores = await state.ready()
    rate = await enforce_rate(request, QUERY_LIMITS)

    api_key = extract_key(request)
    if not api_key:
        raise HTTPException(401, "Add your own API key (Gemini, Groq or OpenRouter) to run a live query.")
    choice = ModelChoice(provider=req.model.provider, model_id=req.model.model_id, api_key=api_key)

    if not settings.mock_llm:
        verdict = await state.keys.check(choice.provider, api_key)
        if verdict is False:
            raise HTTPException(401, f"{choice.provider} rejected that API key. Check it and try again.")

    who = client_id(request)
    slot, reason = state.gate.try_acquire(who)
    if slot is None:
        if reason == "client":
            raise HTTPException(
                429,
                f"You already have {settings.max_runs_per_client} questions running. Wait for one to finish.",
                headers={"Retry-After": "10"},
            )
        raise HTTPException(503, "PaperGraph is at capacity right now. Try again shortly.", headers={"Retry-After": "15"})

    tools = build_toolset(stores, workspaces_for(req.session_id))
    history = _truncate_history(req.history)

    async def event_stream():
        recorder = RunRecorder(req.message, choice)

        def emit(event: str, data: dict) -> dict:
            recorder.add(event, data)
            return _sse(event, data)

        try:
            if settings.mock_llm:
                from mock_run import mock_events

                async with asyncio.timeout(settings.run_timeout_s):
                    async for event, data in mock_events(tools, req.message):
                        yield emit(event, data)
                yield emit("done", {"errored": False})
                return

            try:
                team = build_papergraph_team(tools, choice, history)
            except MissingKeyError as e:
                yield _sse("step", _error_step("Missing key", str(e)))
                yield _sse("done", {"errored": True})
                return

            builder = RunTraceBuilder()
            async with asyncio.timeout(settings.run_timeout_s):
                async with contextlib.aclosing(
                    team.arun(req.message, stream=True, stream_events=True)
                ) as events:
                    async for item in events:
                        steps, deltas = builder.feed(item)
                        for delta in deltas:
                            yield emit("delta", {"text": delta})
                        for step in steps:
                            data = step.to_dict()
                            if step.kind == "error":
                                yield _sse("step", _error_step("Provider error", data["body"], api_key))
                                yield _sse("done", {"errored": True})
                                return
                            yield emit("step", data)

            yield emit("answer", {"answer": builder.final_answer, "claims": builder.claims})
            yield emit("done", {"errored": False})
            recorder.save()

        except TimeoutError:
            yield _sse(
                "step",
                _error_step("Run took too long", f"Stopped after {settings.run_timeout_s:.0f}s to protect your quota."),
            )
            yield _sse("done", {"errored": True})
        except asyncio.CancelledError:
            raise  # client went away: let cancellation stop the run
        except Exception as e:
            yield _sse("step", _error_step("Server error", f"{type(e).__name__}: {e}", api_key))
            yield _sse("done", {"errored": True})
        finally:
            slot.release()

    return EventSourceResponse(
        event_stream(),
        ping=15,
        headers={**rate.headers(), "Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        background=BackgroundTask(slot.release),
    )


# ---------------------------------------------------------------------------
# /api/ingest — full profile only
# ---------------------------------------------------------------------------

@app.post("/api/ingest")
async def ingest(req: IngestRequest, request: Request):
    stores = await state.ready()
    if not settings.enable_ingest or stores.graph.read_only:
        raise HTTPException(
            403,
            "Uploads are disabled on the public demo (it serves a read-only snapshot). "
            "Run the full stack with docker compose to ingest your own papers.",
        )
    await enforce_rate(request, INGEST_LIMITS)
    api_key = extract_key(request)
    if not api_key:
        raise HTTPException(401, "Ingestion runs concept extraction on your own key — add one first.")
    choice = ModelChoice(provider=req.model.provider, model_id=req.model.model_id, api_key=api_key)

    async def stage_stream():
        from etl import IngestError, ingest_arxiv_id

        try:
            result = await ingest_arxiv_id(req.value, req.session_id, choice, stores)
            yield _sse("complete", result)
        except IngestError as e:
            yield _sse("error", {"message": redact(str(e), api_key)[:500]})
        except Exception as e:
            yield _sse("error", {"message": redact(f"{type(e).__name__}: {e}", api_key)[:500]})

    return EventSourceResponse(stage_stream(), ping=15)


# ---------------------------------------------------------------------------
# /api/models/{provider}
# ---------------------------------------------------------------------------

@app.post("/api/models/{provider}")
async def list_models(provider: str, request: Request):
    if provider not in PROVIDERS:
        raise HTTPException(400, f"Unknown provider: {provider}")
    await enforce_rate(request, MODELS_LIMITS)
    key = extract_key(request)

    if provider == "openrouter":
        # Public endpoint, same for everyone — cache it so N visitors cost
        # one upstream call per 10 minutes, not N.
        cached = state._models_cache.get("openrouter")
        if cached and cached[0] > time.monotonic():
            return {"models": cached[1]}
        resp = await state.http.get("https://openrouter.ai/api/v1/models")
        resp.raise_for_status()
        models = [{"id": m["id"], "name": m.get("name", m["id"])} for m in resp.json().get("data", [])]
        state._models_cache["openrouter"] = (time.monotonic() + 600, models)
        return {"models": models}

    if not key:
        return {"models": [], "requires_key": True}
    try:
        if provider == "groq":
            resp = await state.http.get(
                "https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {key}"}
            )
            resp.raise_for_status()
            return {
                "models": [
                    {"id": m["id"], "name": m["id"]}
                    for m in resp.json().get("data", [])
                    if not any(x in m["id"].lower() for x in _GROQ_NON_CHAT_MARKERS)
                ]
            }
        resp = await state.http.get(
            "https://generativelanguage.googleapis.com/v1beta/models", headers={"x-goog-api-key": key}
        )
        resp.raise_for_status()
        return {
            "models": [
                {"id": m["name"].removeprefix("models/"), "name": m.get("displayName", m["name"])}
                for m in resp.json().get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])
            ]
        }
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, f"{provider} refused the model list request.")


# ---------------------------------------------------------------------------
# Corpus reads
# ---------------------------------------------------------------------------

async def _view_for(session_id: Optional[str]) -> tuple[dict, bool]:
    """(graph view, is_shared_only). Shared-only views are precomputed once."""
    stores = await state.ready()
    if session_id and valid_session_id(session_id) and not stores.graph.read_only:
        view = await build_graph_view(stores.graph, workspaces_for(session_id))
        if any(p["own"] for p in view["papers"]):
            return view, False
    return state.shared_view or {"papers": [], "concepts": [], "pos": {}}, True


def _cache_headers(shared: bool) -> dict:
    return {"Cache-Control": "public, max-age=300" if shared else "private, no-store"}


@app.get("/api/papers")
async def list_papers(request: Request, session_id: Optional[str] = None):
    await enforce_rate(request, READ_LIMITS)
    view, shared = await _view_for(session_id)
    return JSONResponse(papers_payload(view), headers=_cache_headers(shared))


@app.get("/api/graph")
async def get_graph(request: Request, session_id: Optional[str] = None):
    await enforce_rate(request, READ_LIMITS)
    view, shared = await _view_for(session_id)
    return JSONResponse(view, headers=_cache_headers(shared))


@app.get("/api/config")
async def get_config():
    stores = await state.ready()
    return {
        "profile": stores.profile,
        "require_byok": settings.require_byok,
        "ingest_enabled": bool(settings.enable_ingest and not stores.graph.read_only),
        "providers": list(PROVIDERS),
        "mock_llm": settings.mock_llm,
        "limits": {
            "query_per_min": settings.query_limit_per_min,
            "query_per_hour": settings.query_limit_per_hour,
            "max_runs_per_client": settings.max_runs_per_client,
            "run_timeout_s": settings.run_timeout_s,
        },
        "corpus": {k: v for k, v in stores.info.items() if k in ("papers", "chunks", "snapshot_exported_at")},
    }


@app.get("/api/health")
async def health():
    try:
        stores = await state.ready()
    except Exception as e:
        return JSONResponse({"status": "starting", "error": type(e).__name__}, status_code=503)
    return {
        "status": "ok",
        "profile": stores.profile,
        "runs_in_flight": state.gate.in_flight,
        "rate_limit_backend": state.rate_backend,
        "uptime_s": round(time.time() - state.started_at),
        "init": {k: v for k, v in stores.info.items() if k in ("load_ms", "chunks", "papers")},
    }


# Static frontend for container deployments (Vercel serves it from its CDN).
if settings.serve_frontend and (ROOT / "frontend").is_dir():
    app.mount("/", StaticFiles(directory=str(ROOT / "frontend"), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
