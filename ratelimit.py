"""
ratelimit.py — request-rate limits and run-concurrency limits
=============================================================

Two different problems, two different tools:

1. RATE (how many requests per client per window) — `RateLimiter`
   Sliding-window counter (the approximation Cloudflare and Upstash use):

       estimate = prev_window_count * (1 - elapsed/window) + curr_window_count

   Two integers per (client, limit), O(1) per request, no per-request
   timestamps, and no 2x burst at window boundaries (fixed windows allow
   `limit` at 0:59 and again at 1:00).

   Backends:
     * UpstashLimiter — Redis over HTTPS (works from serverless; no TCP
       pool to manage). Counters are shared by every instance, so a
       client can't multiply its quota by landing on different instances.
     * MemoryLimiter — per-instance dict. Used for local dev and as the
       fail-open fallback if Redis is unreachable: a Redis outage degrades
       limits to per-instance instead of taking the API down.
   Rejected requests are not counted, so a client hammering a limit
   doesn't extend its own lockout indefinitely.

2. CONCURRENCY (how many agent runs are in flight) — `ConcurrencyGate`
   A run holds an SSE connection for 30-120 s. Rate limits alone don't
   bound in-flight work (6/min/IP × many IPs can still pile up), so each
   instance also caps total runs and runs per client. Over the cap we shed
   load immediately (503/429 + Retry-After) instead of queueing: a queued
   SSE request just becomes a timeout later, after holding a socket.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import httpx


@dataclass(frozen=True)
class Limit:
    name: str  # e.g. "query:min"
    limit: int
    window_s: int


@dataclass
class Decision:
    allowed: bool
    limit: Limit
    remaining: int
    retry_after_s: int

    def headers(self) -> dict[str, str]:
        h = {
            "X-RateLimit-Limit": str(self.limit.limit),
            "X-RateLimit-Remaining": str(max(self.remaining, 0)),
            "X-RateLimit-Policy": f"{self.limit.limit};w={self.limit.window_s}",
        }
        if not self.allowed:
            h["Retry-After"] = str(self.retry_after_s)
        return h


def _estimate(prev: int, curr: int, elapsed: float, window: int) -> float:
    return prev * max(0.0, 1.0 - elapsed / window) + curr


def _retry_after(prev: int, curr: int, elapsed: float, window: int, limit: int) -> int:
    """Seconds until one more request would fit under the limit."""
    if curr + 1 > limit:
        return max(1, math.ceil(window - elapsed))  # must wait for the next window
    if prev <= 0:
        return 1
    # prev*(1 - (elapsed+t)/w) + curr + 1 <= limit  ->  solve for t
    t = window * (1 - (limit - curr - 1) / prev) - elapsed
    return max(1, math.ceil(t))


class Backend(Protocol):
    async def hit(self, key: str, limit: Limit, now: Optional[float] = None) -> Decision: ...
    async def undo(self, key: str, limit: Limit, now: Optional[float] = None) -> None: ...


class MemoryLimiter:
    """Single event loop => no locking needed. Bounded LRU so a flood of
    distinct client ids can't grow memory without limit."""

    def __init__(self, max_keys: int = 100_000):
        self._data: "OrderedDict[str, tuple[int, int, int]]" = OrderedDict()  # key -> (window_idx, prev, curr)
        self._max = max_keys

    def _state(self, key: str, limit: Limit, now: float) -> tuple[str, int, int, int, float]:
        idx = int(now // limit.window_s)
        elapsed = now - idx * limit.window_s
        k = f"{limit.name}|{key}"
        widx, prev, curr = self._data.get(k, (idx, 0, 0))
        if widx == idx - 1:
            prev, curr = curr, 0
        elif widx < idx - 1:
            prev, curr = 0, 0
        return k, idx, prev, curr, elapsed

    async def hit(self, key: str, limit: Limit, now: Optional[float] = None) -> Decision:
        now = time.time() if now is None else now
        k, idx, prev, curr, elapsed = self._state(key, limit, now)
        if _estimate(prev, curr + 1, elapsed, limit.window_s) > limit.limit:
            self._data[k] = (idx, prev, curr)
            return Decision(False, limit, 0, _retry_after(prev, curr, elapsed, limit.window_s, limit.limit))
        self._data[k] = (idx, prev, curr + 1)
        self._data.move_to_end(k)
        while len(self._data) > self._max:
            self._data.popitem(last=False)
        remaining = int(limit.limit - _estimate(prev, curr + 1, elapsed, limit.window_s))
        return Decision(True, limit, remaining, 0)

    async def undo(self, key: str, limit: Limit, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        k, idx, prev, curr, _ = self._state(key, limit, now)
        self._data[k] = (idx, prev, max(curr - 1, 0))


class UpstashLimiter:
    """Same algorithm, counters in Upstash Redis via its REST pipeline API.
    One round trip per check: INCR current window, set its TTL, GET previous."""

    def __init__(self, url: str, token: str, fallback: Optional[MemoryLimiter] = None, timeout_s: float = 1.5):
        self._url = url.rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=timeout_s, headers={"Authorization": f"Bearer {token}"}
        )
        self._fallback = fallback or MemoryLimiter()
        self._warned = False

    async def _pipeline(self, commands: list[list]) -> list:
        resp = await self._client.post(f"{self._url}/pipeline", json=commands)
        resp.raise_for_status()
        out = resp.json()
        errors = [r["error"] for r in out if isinstance(r, dict) and r.get("error")]
        if errors:
            raise RuntimeError(f"upstash: {errors[0]}")
        return [r.get("result") if isinstance(r, dict) else r for r in out]

    def _keys(self, key: str, limit: Limit, now: float) -> tuple[str, str, float]:
        idx = int(now // limit.window_s)
        base = f"pg:rl:{limit.name}:{key}"
        return f"{base}:{idx}", f"{base}:{idx - 1}", now - idx * limit.window_s

    async def hit(self, key: str, limit: Limit, now: Optional[float] = None) -> Decision:
        now = time.time() if now is None else now
        curr_k, prev_k, elapsed = self._keys(key, limit, now)
        try:
            curr, _, prev = await self._pipeline(
                [["INCR", curr_k], ["EXPIRE", curr_k, limit.window_s * 2], ["GET", prev_k]]
            )
            curr, prev = int(curr), int(prev or 0)
        except Exception as e:  # fail open to per-instance limits, never fail closed
            if not self._warned:
                print(f"[ratelimit] Upstash unavailable ({type(e).__name__}); using per-instance limits")
                self._warned = True
            return await self._fallback.hit(key, limit, now)
        if _estimate(prev, curr, elapsed, limit.window_s) > limit.limit:
            try:
                await self._pipeline([["DECR", curr_k]])
            except Exception:
                pass
            return Decision(False, limit, 0, _retry_after(prev, curr - 1, elapsed, limit.window_s, limit.limit))
        return Decision(True, limit, int(limit.limit - _estimate(prev, curr, elapsed, limit.window_s)), 0)

    async def undo(self, key: str, limit: Limit, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        curr_k, _, _ = self._keys(key, limit, now)
        try:
            await self._pipeline([["DECR", curr_k]])
        except Exception:
            await self._fallback.undo(key, limit, now)

    async def close(self) -> None:
        await self._client.aclose()


class RateLimiter:
    def __init__(self, backend: Backend):
        self.backend = backend

    async def check(self, key: str, limits: Sequence[Limit]) -> Decision:
        """All limits must pass. If a later one fails, earlier increments are
        rolled back so the rejected request costs the client nothing.
        Returns the tightest decision (for headers)."""
        passed: list[Limit] = []
        tightest: Optional[Decision] = None
        for lim in limits:
            d = await self.backend.hit(key, lim)
            if not d.allowed:
                for p in passed:
                    await self.backend.undo(key, p)
                return d
            passed.append(lim)
            if tightest is None or d.remaining < tightest.remaining:
                tightest = d
        assert tightest is not None
        return tightest


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

class Slot:
    def __init__(self, gate: "ConcurrencyGate", client: str):
        self._gate = gate
        self._client = client
        self._released = False

    def release(self) -> None:
        """Idempotent — called from the stream's finally AND the response's
        background task, so a client that disconnects before the first byte
        can't leak a slot."""
        if self._released:
            return
        self._released = True
        self._gate._release(self._client)


class ConcurrencyGate:
    def __init__(self, max_total: int, max_per_client: int):
        self.max_total = max_total
        self.max_per_client = max_per_client
        self.in_flight = 0
        self._per_client: dict[str, int] = {}

    def try_acquire(self, client: str) -> tuple[Optional[Slot], Optional[str]]:
        if self.in_flight >= self.max_total:
            return None, "capacity"
        if self._per_client.get(client, 0) >= self.max_per_client:
            return None, "client"
        self.in_flight += 1
        self._per_client[client] = self._per_client.get(client, 0) + 1
        return Slot(self, client), None

    def _release(self, client: str) -> None:
        self.in_flight = max(self.in_flight - 1, 0)
        n = self._per_client.get(client, 0) - 1
        if n <= 0:
            self._per_client.pop(client, None)
        else:
            self._per_client[client] = n
