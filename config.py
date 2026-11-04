"""
config.py — one place for every runtime knob
============================================

Everything is driven by environment variables so the same code runs in
three very different places without edits:

  * STORE_PROFILE=demo  (default) — the public, zero-cost deployment
      (Vercel / Cloud Run / any container). Qdrant runs *embedded* in the
      process and the knowledge graph is served from a snapshot that was
      exported from Neo4j. No external database, nothing to pause or
      expire, cold start is bounded by process start + snapshot load.

  * STORE_PROFILE=full  — the system of record. Talks to a real Qdrant
      server and a real Neo4j server (docker compose locally, or Qdrant
      Cloud / AuraDB when you want hosted). This is where ingestion runs
      and where the demo snapshot is exported from.

  * Tests / CI           — same code, with the hashing embedder and
      Qdrant/Neo4j service containers.

Read once at import time into an immutable Settings object. Nothing in
here is a secret except the optional database / Upstash credentials —
LLM keys are never configured server-side in demo mode (BYOK only).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # .env for local dev; real env vars still win

ROOT = Path(__file__).resolve().parent


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else default


def _list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    # --- storage profile -------------------------------------------------
    store_profile: str = field(default_factory=lambda: os.environ.get("STORE_PROFILE", "demo").lower())
    snapshot_dir: Path = field(default_factory=lambda: Path(os.environ.get("SNAPSHOT_DIR", str(ROOT / "snapshot"))))

    # Qdrant (full profile). In demo profile Qdrant runs in-process.
    qdrant_url: str = field(default_factory=lambda: os.environ.get("QDRANT_URL", "http://localhost:6333"))
    qdrant_api_key: str | None = field(default_factory=lambda: os.environ.get("QDRANT_API_KEY") or None)
    qdrant_collection: str = field(default_factory=lambda: os.environ.get("QDRANT_COLLECTION", "papergraph_chunks_v2"))

    # Neo4j (full profile). AuraDB's credential file says NEO4J_USERNAME.
    neo4j_uri: str = field(default_factory=lambda: os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(
        default_factory=lambda: os.environ.get("NEO4J_USERNAME") or os.environ.get("NEO4J_USER", "neo4j")
    )
    neo4j_password: str = field(default_factory=lambda: os.environ.get("NEO4J_PASSWORD", "papergraph"))
    neo4j_database: str | None = field(default_factory=lambda: os.environ.get("NEO4J_DATABASE") or None)

    # --- embeddings ------------------------------------------------------
    # "fastembed" (ONNX, no torch) in every real deployment; "hashing" is a
    # deterministic, dependency-free stand-in used by tests and CI.
    embedder: str = field(default_factory=lambda: os.environ.get("EMBEDDER", "fastembed").lower())
    dense_model: str = field(default_factory=lambda: os.environ.get("DENSE_MODEL", "BAAI/bge-small-en-v1.5"))
    dense_dim: int = field(default_factory=lambda: _int("DENSE_DIM", 384))
    sparse_model: str = field(default_factory=lambda: os.environ.get("SPARSE_MODEL", "Qdrant/bm25"))
    # Serverless filesystems are read-only except /tmp (Vercel sets VERCEL=1).
    model_cache_dir: str | None = field(
        default_factory=lambda: os.environ.get("MODEL_CACHE_DIR")
        or ("/tmp/fastembed_cache" if os.environ.get("VERCEL") else None)
    )
    enable_reranker: bool = field(default_factory=lambda: _bool("ENABLE_RERANKER", False))
    rerank_model: str = field(
        default_factory=lambda: os.environ.get("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
    )
    embed_workers: int = field(default_factory=lambda: _int("EMBED_WORKERS", 4))

    # --- product behaviour -------------------------------------------------
    # BYOK: every live run uses the *visitor's* key. When true the server
    # never falls back to its own provider keys, so the owner pays nothing
    # per request no matter how many people hit the demo.
    require_byok: bool = field(default_factory=lambda: _bool("REQUIRE_BYOK", True))
    enable_ingest: bool = field(
        default_factory=lambda: _bool("ENABLE_INGEST", os.environ.get("STORE_PROFILE", "demo").lower() == "full")
    )
    serve_frontend: bool = field(default_factory=lambda: _bool("SERVE_FRONTEND", True))
    cors_origins: list[str] = field(default_factory=lambda: _list("CORS_ORIGINS", ["*"]))
    mock_llm: bool = field(default_factory=lambda: _bool("MOCK_LLM", False))
    record_runs_dir: str | None = field(default_factory=lambda: os.environ.get("RECORD_RUNS_DIR") or None)

    # --- abuse / capacity controls -----------------------------------------
    # Rate limits are sliding-window counters (see ratelimit.py). Backed by
    # Upstash Redis when UPSTASH_* is set (shared across every instance),
    # otherwise per-instance memory.
    upstash_url: str | None = field(default_factory=lambda: os.environ.get("UPSTASH_REDIS_REST_URL") or None)
    upstash_token: str | None = field(default_factory=lambda: os.environ.get("UPSTASH_REDIS_REST_TOKEN") or None)
    query_limit_per_min: int = field(default_factory=lambda: _int("QUERY_LIMIT_PER_MIN", 6))
    query_limit_per_hour: int = field(default_factory=lambda: _int("QUERY_LIMIT_PER_HOUR", 40))
    ingest_limit_per_hour: int = field(default_factory=lambda: _int("INGEST_LIMIT_PER_HOUR", 3))
    models_limit_per_min: int = field(default_factory=lambda: _int("MODELS_LIMIT_PER_MIN", 20))
    read_limit_per_min: int = field(default_factory=lambda: _int("READ_LIMIT_PER_MIN", 120))
    # Load shedding: an agent run holds an SSE connection for 30-120s, so
    # queueing excess work only converts it into timeouts. Past these caps
    # we answer 503/429 immediately with Retry-After instead.
    max_concurrent_runs: int = field(default_factory=lambda: _int("MAX_CONCURRENT_RUNS", 40))
    max_runs_per_client: int = field(default_factory=lambda: _int("MAX_RUNS_PER_CLIENT", 2))
    run_timeout_s: float = field(default_factory=lambda: _float("RUN_TIMEOUT_S", 150.0))
    agent_tool_call_limit: int = field(default_factory=lambda: _int("AGENT_TOOL_CALL_LIMIT", 6))
    team_tool_call_limit: int = field(default_factory=lambda: _int("TEAM_TOOL_CALL_LIMIT", 10))
    trust_proxy_headers: bool = field(default_factory=lambda: _bool("TRUST_PROXY_HEADERS", True))
    key_check_ttl_s: int = field(default_factory=lambda: _int("KEY_CHECK_TTL_S", 600))

    # --- request size limits -------------------------------------------------
    max_message_chars: int = 2000
    max_history_turns: int = 6
    max_history_chars: int = 4000

    @property
    def is_demo(self) -> bool:
        return self.store_profile != "full"


settings = Settings()

SHARED_WORKSPACE = "shared"  # the base corpus every session can see
