"""
Test configuration.

Tests never download models or call an LLM: EMBEDDER=hashing swaps in a
deterministic feature-hashing embedder, and MOCK_LLM runs are scripted.

Backends that need a server are opt-in by reachability:
  QDRANT_URL (default http://localhost:6333) and NEO4J_URI
  (default bolt://localhost:7687, auth neo4j/papergraph). CI provides
  both as service containers; locally `docker compose up qdrant neo4j`.
If a server isn't reachable its parametrized cases are skipped, never
silently passed.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("EMBEDDER", "hashing")
os.environ.setdefault("STORE_PROFILE", "demo")
os.environ.setdefault("RATE_LIMIT_BACKEND", "memory")
os.environ.pop("UPSTASH_REDIS_REST_URL", None)
os.environ.pop("UPSTASH_REDIS_REST_TOKEN", None)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
