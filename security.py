"""
security.py — client identity, BYOK key handling, redaction
===========================================================

Rules this module enforces for visitors' LLM keys:

  * Keys travel only in the `X-LLM-Api-Key` request header — never in a
    URL (v1's GET /models?api_key=... put keys into access logs, browser
    history and proxy logs) and never in a JSON body that might be logged.
  * Keys are never logged, persisted, or echoed back. Provider error
    messages are passed through `redact()` before reaching the client or
    a log line, because SDK errors sometimes include the request URL.
  * A key is checked against the provider's cheapest authenticated
    endpoint *before* a run starts, so a typo costs one fast 401 instead
    of a 5-agent team spinning up and failing two steps in. Results are
    cached under a SHA-256 of the key (the key itself is not the cache
    key), valid keys for KEY_CHECK_TTL_S, invalid ones for 60 s.
  * If the provider can't be reached (timeout, 5xx, 429) the check
    fails OPEN — the run itself will surface a real error — so a slow
    provider status endpoint never blocks valid users.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import OrderedDict
from typing import Optional

import httpx
from fastapi import Request

from config import SHARED_WORKSPACE, settings

KEY_HEADER = "x-llm-api-key"

_SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_KEY_RE = re.compile(r"^[\x21-\x7e]{8,300}$")  # printable ASCII, no whitespace

_SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),  # Google
    re.compile(r"gsk_[0-9A-Za-z]{20,}"),  # Groq
    re.compile(r"sk-or-[0-9A-Za-z_\-]{20,}"),  # OpenRouter
    re.compile(r"sk-[0-9A-Za-z_\-]{20,}"),  # OpenAI-style
    re.compile(r"(?i)([?&](?:key|api_key|token)=)[^&\s\"']+"),
    re.compile(r"(?i)(bearer\s+)[0-9A-Za-z._\-]{12,}"),
]


def valid_session_id(session_id: str) -> bool:
    return bool(_SESSION_RE.match(session_id or "")) and session_id != SHARED_WORKSPACE


def workspaces_for(session_id: str) -> list[str]:
    return [SHARED_WORKSPACE, session_id]


def client_id(request: Request) -> str:
    """Best-effort client identity for rate limiting.

    Behind a platform proxy the socket peer is the proxy, so we read the
    header that platform sets: Vercel overwrites x-real-ip /
    x-vercel-forwarded-for with the true client address. A generic
    X-Forwarded-For's first hop is client-controlled on some proxies, which
    is why the per-instance concurrency cap exists as a second line that
    doesn't depend on identity at all.
    """
    if settings.trust_proxy_headers:
        for header in ("x-vercel-forwarded-for", "x-real-ip", "fly-client-ip", "cf-connecting-ip"):
            value = request.headers.get(header)
            if value:
                return value.split(",")[0].strip()
        xff = request.headers.get("x-forwarded-for")
        if xff:
            return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def extract_key(request: Request) -> Optional[str]:
    key = (request.headers.get(KEY_HEADER) or "").strip()
    return key if key and _KEY_RE.match(key) else None


def redact(text: str, *secrets: Optional[str]) -> str:
    if not text:
        return text
    for s in secrets:
        if s and len(s) >= 6:
            text = text.replace(s, "[REDACTED]")
    for pat in _SECRET_PATTERNS:
        text = pat.sub(lambda m: (m.group(1) if m.groups() else "") + "[REDACTED]", text)
    return text


def key_fingerprint(provider: str, key: str) -> str:
    return hashlib.sha256(f"{provider}:{key}".encode()).hexdigest()


class KeyChecker:
    """Validates a provider key with one cheap authenticated GET."""

    def __init__(self, client: Optional[httpx.AsyncClient] = None, ttl_s: int = 600, max_entries: int = 10_000):
        self._client = client or httpx.AsyncClient(timeout=4.0)
        self._ttl = ttl_s
        self._cache: "OrderedDict[str, tuple[bool, float]]" = OrderedDict()
        self._max = max_entries

    async def close(self) -> None:
        await self._client.aclose()

    def _request(self, provider: str, key: str) -> tuple[str, dict]:
        if provider == "gemini":
            return "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1", {"x-goog-api-key": key}
        if provider == "groq":
            return "https://api.groq.com/openai/v1/models", {"Authorization": f"Bearer {key}"}
        if provider == "openrouter":
            return "https://openrouter.ai/api/v1/key", {"Authorization": f"Bearer {key}"}
        raise ValueError(f"unknown provider {provider!r}")

    async def check(self, provider: str, key: str) -> Optional[bool]:
        """True = valid, False = rejected by provider, None = couldn't tell."""
        fp = key_fingerprint(provider, key)
        hit = self._cache.get(fp)
        now = time.monotonic()
        if hit and hit[1] > now:
            return hit[0]
        url, headers = self._request(provider, key)
        try:
            resp = await self._client.get(url, headers=headers)
        except httpx.HTTPError:
            return None
        if resp.status_code in (401, 403) or (provider == "gemini" and resp.status_code == 400):
            verdict = False
        elif resp.is_success:
            verdict = True
        else:
            return None  # 429 / 5xx: inconclusive, fail open
        self._cache[fp] = (verdict, now + (self._ttl if verdict else 60))
        self._cache.move_to_end(fp)
        while len(self._cache) > self._max:
            self._cache.popitem(last=False)
        return verdict
