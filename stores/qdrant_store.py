"""
stores/qdrant_store.py — hybrid chunk search on Qdrant
======================================================

One class, two deployment shapes:

  * full profile : AsyncQdrantClient(url=..., api_key=...) -> Qdrant server
  * demo profile : AsyncQdrantClient(location=":memory:")  -> Qdrant embedded
                   in the API process, loaded from snapshot/chunks.jsonl.gz

Both run the exact same collection schema and the exact same query:

    prefetch dense (bge-small, cosine)  ┐
                                        ├─ Reciprocal Rank Fusion ─> (optional
    prefetch sparse (BM25, IDF)         ┘                             cross-encoder rerank)

with the workspace filter applied inside *each* prefetch, so a chunk
from another session's upload can never enter the candidate pool.
The filter is built from the `workspaces` argument the server passes —
the LLM-facing tool has no parameter that could widen it.

Point ids are uuid5(workspace, paper, chunk_index): re-ingesting a paper
overwrites its chunks instead of duplicating them.
"""

from __future__ import annotations

import uuid
from typing import Iterable, Optional, Sequence

from qdrant_client import AsyncQdrantClient, models

from config import SHARED_WORKSPACE
from embeddings import Embedder, Reranker, run_in_pool
from stores.base import ChunkHit, ChunkRecord, base_arxiv_id, normalize_paper_ref

DENSE = "dense"
SPARSE = "bm25"
_NAMESPACE = uuid.UUID("6f1c2a52-6b8e-4d7a-9a51-2f7c0b0c9e11")
KEYWORD_INDEXES = ("workspace_id", "paper_base_id", "section")


def point_id(workspace_id: str, paper_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{workspace_id}|{paper_id}|{chunk_index}"))


class QdrantVectorStore:
    def __init__(
        self,
        client: AsyncQdrantClient,
        collection: str,
        embedder: Embedder,
        reranker: Optional[Reranker] = None,
        embedded: bool = False,
    ):
        self.client = client
        self.collection = collection
        self.embedder = embedder
        self.reranker = reranker
        self.embedded = embedded

    # -- lifecycle -----------------------------------------------------------

    async def ensure_ready(self) -> None:
        if not await self.client.collection_exists(self.collection):
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config={DENSE: models.VectorParams(size=self.embedder.dim, distance=models.Distance.COSINE)},
                sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )
        if not self.embedded:
            # Qdrant server rejects filtered queries on un-indexed payload
            # fields (400), so index every field we filter on. Idempotent.
            for field in KEYWORD_INDEXES:
                try:
                    await self.client.create_payload_index(
                        collection_name=self.collection,
                        field_name=field,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    pass  # already exists

    async def close(self) -> None:
        await self.client.close()

    async def count(self) -> int:
        if not await self.client.collection_exists(self.collection):
            return 0
        return (await self.client.count(self.collection, exact=True)).count

    # -- write path (ETL / ingest) ------------------------------------------------

    async def upsert_chunks(self, chunks: Sequence[ChunkRecord]) -> int:
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        dense = await run_in_pool(self.embedder.embed_documents, texts)
        sparse = await run_in_pool(self.embedder.sparse_documents, texts)
        points = [
            models.PointStruct(
                id=point_id(c.workspace_id, c.paper_id, c.chunk_index),
                vector={
                    DENSE: d,
                    SPARSE: models.SparseVector(indices=list(s.indices), values=list(s.values)),
                },
                payload={
                    "text": c.text,
                    "paper_id": c.paper_id,
                    "paper_base_id": base_arxiv_id(c.paper_id),
                    "title": c.title,
                    "section": c.section,
                    "chunk_index": c.chunk_index,
                    "workspace_id": c.workspace_id,
                    "arxiv_url": c.arxiv_url,
                },
            )
            for c, d, s in zip(chunks, dense, sparse)
        ]
        for start in range(0, len(points), 128):
            await self.client.upsert(self.collection, points=points[start : start + 128], wait=True)
        return len(points)

    async def export_points(self, workspaces: Sequence[str] = (SHARED_WORKSPACE,)) -> list[dict]:
        """Every point with its vectors, for the demo snapshot. Shared
        corpus only by default: visitors' uploads never leave the server."""
        out: list[dict] = []
        offset = None
        flt = models.Filter(
            must=[models.FieldCondition(key="workspace_id", match=models.MatchAny(any=list(workspaces)))]
        )
        while True:
            batch, offset = await self.client.scroll(
                self.collection,
                limit=256,
                offset=offset,
                scroll_filter=flt,
                with_payload=True,
                with_vectors=True,
            )
            for p in batch:
                sparse = p.vector[SPARSE]
                out.append(
                    {
                        "id": str(p.id),
                        "payload": p.payload,
                        "dense": [round(float(x), 6) for x in p.vector[DENSE]],
                        "sparse": {"indices": list(sparse.indices), "values": list(sparse.values)},
                    }
                )
            if offset is None:
                return out

    async def import_points(self, points: Iterable[dict]) -> int:
        """Load pre-computed points (no embedding work at startup)."""
        batch: list[models.PointStruct] = []
        n = 0
        for p in points:
            batch.append(
                models.PointStruct(
                    id=p["id"],
                    payload=p["payload"],
                    vector={
                        DENSE: p["dense"],
                        SPARSE: models.SparseVector(**p["sparse"]),
                    },
                )
            )
            if len(batch) >= 256:
                await self.client.upsert(self.collection, points=batch, wait=True)
                n += len(batch)
                batch = []
        if batch:
            await self.client.upsert(self.collection, points=batch, wait=True)
            n += len(batch)
        return n

    # -- read path ------------------------------------------------------------

    @staticmethod
    def build_filter(
        workspaces: Sequence[str],
        paper_id: Optional[str] = None,
        sections: Optional[Sequence[str]] = None,
    ) -> models.Filter:
        if not workspaces:
            raise ValueError("workspaces must be non-empty — refusing an unscoped search")
        must: list[models.Condition] = [
            models.FieldCondition(key="workspace_id", match=models.MatchAny(any=list(workspaces)))
        ]
        if paper_id:
            must.append(
                models.FieldCondition(key="paper_base_id", match=models.MatchValue(value=normalize_paper_ref(paper_id)))
            )
        if sections:
            must.append(models.FieldCondition(key="section", match=models.MatchAny(any=list(sections))))
        return models.Filter(must=must)

    async def search(
        self,
        query: str,
        workspaces: Sequence[str],
        k: int = 6,
        paper_id: Optional[str] = None,
        sections: Optional[Sequence[str]] = None,
    ) -> list[ChunkHit]:
        flt = self.build_filter(workspaces, paper_id, sections)
        dense = await run_in_pool(self.embedder.embed_query, query)
        sparse = await run_in_pool(self.embedder.sparse_query, query)

        fetch = k * 3 if self.reranker else k
        prefetch_n = max(fetch * 4, 20)
        prefetch = [models.Prefetch(query=dense, using=DENSE, limit=prefetch_n, filter=flt)]
        if sparse.indices:
            prefetch.append(
                models.Prefetch(
                    query=models.SparseVector(indices=list(sparse.indices), values=list(sparse.values)),
                    using=SPARSE,
                    limit=prefetch_n,
                    filter=flt,
                )
            )
        res = await self.client.query_points(
            self.collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=fetch,
            with_payload=True,
        )
        hits = [
            ChunkHit(
                paper_id=p.payload.get("paper_id", ""),
                title=p.payload.get("title", ""),
                section=p.payload.get("section", ""),
                text=p.payload.get("text", ""),
                score=float(p.score),
            )
            for p in res.points
        ]
        if self.reranker and hits:
            scores = await run_in_pool(self.reranker.rerank, query, [h.text for h in hits])
            for h, s in zip(hits, scores):
                h.score = s
            hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]
