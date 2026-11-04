"""
stores — profile-driven construction of the vector + graph stores
=================================================================

    STORE_PROFILE=demo  -> Qdrant embedded (in-process) + snapshot graph
    STORE_PROFILE=full  -> Qdrant server              + Neo4j server

`open_stores()` is called once per process (FastAPI lifespan, or the ETL
CLI) and the resulting `Stores` object is shared by every request.

Snapshot layout (written by `python etl.py export`, read here):

    snapshot/manifest.json     embedder name/dim, counts, export time
    snapshot/graph.json        GraphSnapshot (papers, concepts, edges)
    snapshot/chunks.jsonl.gz   one Qdrant point per line, vectors included

Vectors are pre-computed at export time, so a cold start does no
embedding work for the corpus — it only parses ~1-3 MB of JSON and
bulk-loads it into the embedded Qdrant.
"""

from __future__ import annotations

import gzip
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from qdrant_client import AsyncQdrantClient

from config import Settings, settings as default_settings
from embeddings import get_embedder, get_reranker
from stores.base import GraphStore, VectorStore
from stores.neo4j_store import Neo4jGraphStore
from stores.qdrant_store import QdrantVectorStore
from stores.snapshot_graph import SnapshotGraphStore

MANIFEST = "manifest.json"
GRAPH_FILE = "graph.json"
CHUNKS_FILE = "chunks.jsonl.gz"


@dataclass
class Stores:
    vector: VectorStore
    graph: GraphStore
    profile: str
    info: dict

    async def close(self) -> None:
        await self.vector.close()
        await self.graph.close()


def iter_snapshot_points(snapshot_dir: Path) -> Iterator[dict]:
    path = snapshot_dir / CHUNKS_FILE
    if not path.exists():
        return
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def read_manifest(snapshot_dir: Path) -> Optional[dict]:
    path = snapshot_dir / MANIFEST
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


async def open_stores(cfg: Settings = default_settings) -> Stores:
    started = time.perf_counter()
    embedder = get_embedder()
    reranker = get_reranker()

    if cfg.store_profile == "full":
        vector = QdrantVectorStore(
            AsyncQdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key, timeout=15),
            cfg.qdrant_collection,
            embedder,
            reranker,
            embedded=False,
        )
        graph = Neo4jGraphStore(cfg.neo4j_uri, cfg.neo4j_user, cfg.neo4j_password, cfg.neo4j_database)
        await vector.ensure_ready()
        await graph.ensure_ready()
        info = {"profile": "full", "qdrant": cfg.qdrant_url, "neo4j": cfg.neo4j_uri}
        return Stores(vector=vector, graph=graph, profile="full", info=info)

    manifest = read_manifest(cfg.snapshot_dir)
    if manifest and manifest.get("dense_dim") != embedder.dim:
        raise RuntimeError(
            f"Snapshot was embedded with {manifest.get('dense_model')} ({manifest.get('dense_dim')}-d) "
            f"but this process uses {embedder.name} ({embedder.dim}-d). Re-export the snapshot "
            f"or set DENSE_MODEL/DENSE_DIM to match."
        )
    vector = QdrantVectorStore(
        AsyncQdrantClient(location=":memory:"),
        cfg.qdrant_collection,
        embedder,
        reranker,
        embedded=True,
    )
    await vector.ensure_ready()
    n_points = await vector.import_points(iter_snapshot_points(cfg.snapshot_dir))
    graph = SnapshotGraphStore.from_file(cfg.snapshot_dir / GRAPH_FILE)
    info = {
        "profile": "demo",
        "chunks": n_points,
        "papers": len(graph.papers),
        "snapshot_exported_at": (manifest or {}).get("exported_at"),
        "load_ms": round((time.perf_counter() - started) * 1000, 1),
    }
    print(f"[stores] demo profile ready: {info}")
    return Stores(vector=vector, graph=graph, profile="demo", info=info)
