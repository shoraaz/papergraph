"""
embeddings.py — dense + sparse embeddings without torch
=======================================================

The previous stack loaded sentence-transformers (and therefore torch,
~2 GB installed) plus a 2.2 GB cross-encoder at startup. That alone made
every cold start take most of a minute and forced ENABLE_RERANKER=false
on 512 MB hosts.

This module swaps that for FastEmbed (ONNX Runtime):

  * dense : BAAI/bge-small-en-v1.5   (384-d, ~67 MB ONNX)
  * sparse: Qdrant/bm25              (stemmer + IDF on the Qdrant side)
  * rerank: Xenova/ms-marco-MiniLM-L-6-v2 (optional, ENABLE_RERANKER)

Same BGE family as before, so retrieval behaviour stays comparable —
but the RAGAS numbers in the README were measured on the old stack and
must be re-run (eval/run_evals.py) before being quoted for this one.

Model inference is CPU-bound, so callers go through `run_in_pool()`,
which uses a small dedicated thread pool (ONNX Runtime releases the GIL)
instead of blocking the event loop or starving asyncio's default
executor. Query embeddings are LRU-cached: the suggested questions in
the UI repeat constantly and should cost nothing after the first hit.

`HashingEmbedder` is a deterministic, dependency-free stand-in used by
tests and CI so they never need to download a model.
"""

from __future__ import annotations

import asyncio
import math
import re
import threading
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional, Protocol, Sequence, TypeVar

from config import settings

T = TypeVar("T")


@dataclass(frozen=True)
class SparseVec:
    indices: tuple[int, ...]
    values: tuple[float, ...]


class Embedder(Protocol):
    dim: int
    name: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
    def sparse_documents(self, texts: Sequence[str]) -> list[SparseVec]: ...
    def sparse_query(self, text: str) -> SparseVec: ...


class Reranker(Protocol):
    def rerank(self, query: str, documents: Sequence[str]) -> list[float]: ...


# ---------------------------------------------------------------------------
# FastEmbed (production)
# ---------------------------------------------------------------------------

class FastEmbedEmbedder:
    """Lazy-loading FastEmbed wrapper. Models load on first use (or on an
    explicit warm() at server startup) and are then shared process-wide."""

    def __init__(self, dense_model: str, dim: int, sparse_model: str, cache_dir: Optional[str] = None):
        self.dim = dim
        self.name = dense_model
        self._dense_model = dense_model
        self._sparse_model = sparse_model
        self._cache_dir = cache_dir
        self._dense = None
        self._sparse = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._dense is not None and self._sparse is not None:
            return
        with self._lock:
            if self._dense is None:
                from fastembed import TextEmbedding

                self._dense = TextEmbedding(model_name=self._dense_model, cache_dir=self._cache_dir)
            if self._sparse is None:
                from fastembed import SparseTextEmbedding

                self._sparse = SparseTextEmbedding(model_name=self._sparse_model, cache_dir=self._cache_dir)

    def warm(self) -> None:
        self._load()
        self.embed_query("warm up")

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self._load()
        return [vec.tolist() for vec in self._dense.embed(list(texts), batch_size=32)]

    @lru_cache(maxsize=2048)
    def embed_query(self, text: str) -> list[float]:
        self._load()
        return next(iter(self._dense.query_embed(text))).tolist()

    def sparse_documents(self, texts: Sequence[str]) -> list[SparseVec]:
        self._load()
        return [
            SparseVec(tuple(int(i) for i in emb.indices), tuple(float(v) for v in emb.values))
            for emb in self._sparse.embed(list(texts), batch_size=32)
        ]

    @lru_cache(maxsize=2048)
    def sparse_query(self, text: str) -> SparseVec:
        self._load()
        emb = next(iter(self._sparse.query_embed(text)))
        return SparseVec(tuple(int(i) for i in emb.indices), tuple(float(v) for v in emb.values))


class FastEmbedReranker:
    def __init__(self, model: str, cache_dir: Optional[str] = None):
        self._model_name = model
        self._cache_dir = cache_dir
        self._model = None
        self._lock = threading.Lock()

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed.rerank.cross_encoder import TextCrossEncoder

                    self._model = TextCrossEncoder(model_name=self._model_name, cache_dir=self._cache_dir)
        return [float(s) for s in self._model.rerank(query, list(documents))]


# ---------------------------------------------------------------------------
# Hashing embedder (tests / CI — no downloads, fully deterministic)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _h(token: str) -> int:
    return zlib.crc32(token.encode("utf-8"))


class HashingEmbedder:
    """Feature-hashing bag of words (+ bigrams). Semantically weak but
    stable, which is exactly what contract tests need: documents sharing
    words with the query score higher, every run, on every machine."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def warm(self) -> None:  # parity with FastEmbedEmbedder
        return None

    def _dense(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        toks = _tokens(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for feat in feats:
            h = _h(feat)
            vec[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _sparse(self, text: str) -> SparseVec:
        counts = Counter(_h(t) % (2**31 - 1) for t in _tokens(text))
        items = sorted(counts.items())
        return SparseVec(tuple(i for i, _ in items), tuple(float(c) for _, c in items))

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._dense(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._dense(text)

    def sparse_documents(self, texts: Sequence[str]) -> list[SparseVec]:
        return [self._sparse(t) for t in texts]

    def sparse_query(self, text: str) -> SparseVec:
        return self._sparse(text)


# ---------------------------------------------------------------------------
# Process-wide singletons + a bounded pool for CPU-bound inference
# ---------------------------------------------------------------------------

_embedder: Optional[Embedder] = None
_reranker: Optional[Reranker] = None
_reranker_resolved = False
_pool: Optional[ThreadPoolExecutor] = None
_singleton_lock = threading.Lock()


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        with _singleton_lock:
            if _embedder is None:
                if settings.embedder == "hashing":
                    _embedder = HashingEmbedder(dim=settings.dense_dim)
                else:
                    _embedder = FastEmbedEmbedder(
                        dense_model=settings.dense_model,
                        dim=settings.dense_dim,
                        sparse_model=settings.sparse_model,
                        cache_dir=settings.model_cache_dir,
                    )
    return _embedder


def set_embedder(embedder: Embedder) -> None:
    """Test hook."""
    global _embedder
    _embedder = embedder


def get_reranker() -> Optional[Reranker]:
    global _reranker, _reranker_resolved
    if not _reranker_resolved:
        _reranker = (
            FastEmbedReranker(settings.rerank_model, cache_dir=settings.model_cache_dir)
            if settings.enable_reranker and settings.embedder != "hashing"
            else None
        )
        _reranker_resolved = True
    return _reranker


def _get_pool() -> ThreadPoolExecutor:
    global _pool
    if _pool is None:
        with _singleton_lock:
            if _pool is None:
                _pool = ThreadPoolExecutor(max_workers=settings.embed_workers, thread_name_prefix="embed")
    return _pool


async def run_in_pool(fn: Callable[..., T], *args) -> T:
    """Run CPU-bound model work off the event loop, in a pool sized for it."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_get_pool(), fn, *args)
