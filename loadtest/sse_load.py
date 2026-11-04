"""
loadtest/sse_load.py — concurrent SSE load against /api/query
=============================================================

Opens N concurrent question streams (each from a distinct simulated
client address via X-Forwarded-For, so per-client limits behave as they
would for N real visitors) and reports:

  * status mix (200 / 429 / 503 …)
  * time to first SSE event (TTFE) and full-run duration percentiles
  * how fast rejected requests were rejected (load shedding should be ms)
  * /api/health latency sampled *while* the streams are open

Run the server in mock mode so you measure the serving path, not an LLM
provider (and don't spend anyone's quota):

    MOCK_LLM=true MAX_CONCURRENT_RUNS=1000 QUERY_LIMIT_PER_MIN=1000 \
      uvicorn server:app --port 8000
    python loadtest/sse_load.py --url http://localhost:8000 --users 500

Spoofing X-Forwarded-For only works because TRUST_PROXY_HEADERS=true and
there is no proxy in front of a local server — which is exactly why
production also has the per-instance concurrency cap that doesn't depend
on client identity.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from collections import Counter

import httpx

QUESTIONS = [
    "What does Self-RAG do?",
    "How does Corrective RAG decide to re-retrieve?",
    "What benchmarks are used in RAG papers?",
    "What should I read before DuetRAG?",
    "Is the DuetRAG improvement claim overstated?",
]


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = min(len(values) - 1, max(0, round(p / 100 * (len(values) - 1))))
    return values[k]


async def one_user(client: httpx.AsyncClient, url: str, i: int, results: list) -> None:
    ip = f"10.{(i >> 16) & 255}.{(i >> 8) & 255}.{i & 255}"
    body = {
        "message": random.choice(QUESTIONS),
        "session_id": f"s_load_{i:08d}",
        "model": {"provider": "groq"},
    }
    headers = {"X-LLM-Api-Key": "gsk_" + "l" * 40, "X-Forwarded-For": ip}
    t0 = time.perf_counter()
    ttfe = None
    status = None
    events = 0
    try:
        async with client.stream("POST", f"{url}/api/query", json=body, headers=headers) as resp:
            status = resp.status_code
            if status == 200:
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        events += 1
                        if ttfe is None:
                            ttfe = time.perf_counter() - t0
            else:
                await resp.aread()
    except Exception as e:
        status = type(e).__name__
    results.append({"status": status, "ttfe": ttfe, "total": time.perf_counter() - t0, "events": events})


async def sample_health(client: httpx.AsyncClient, url: str, stop: asyncio.Event, out: list) -> None:
    while not stop.is_set():
        t0 = time.perf_counter()
        try:
            r = await client.get(f"{url}/api/health")
            out.append((time.perf_counter() - t0, r.json().get("runs_in_flight")))
        except Exception:
            out.append((float("inf"), None))
        await asyncio.sleep(0.25)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--users", type=int, default=200)
    ap.add_argument("--ramp", type=float, default=1.0, help="seconds over which to start all users")
    ap.add_argument("--json", help="write raw summary JSON here")
    args = ap.parse_args()

    limits = httpx.Limits(max_connections=args.users + 20, max_keepalive_connections=args.users + 20)
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0), limits=limits) as client:
        results: list = []
        health: list = []
        stop = asyncio.Event()
        sampler = asyncio.create_task(sample_health(client, args.url, stop, health))
        t0 = time.perf_counter()
        tasks = []
        for i in range(args.users):
            tasks.append(asyncio.create_task(one_user(client, args.url, i, results)))
            if args.ramp:
                await asyncio.sleep(args.ramp / args.users)
        await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0
        stop.set()
        await sampler

    ok = [r for r in results if r["status"] == 200]
    rejected = [r for r in results if r["status"] in (429, 503)]
    summary = {
        "users": args.users,
        "wall_s": round(wall, 2),
        "status": dict(Counter(str(r["status"]) for r in results)),
        "completed_runs_per_s": round(len(ok) / wall, 1),
        "ttfe_ms": {p: round(pct([r["ttfe"] for r in ok if r["ttfe"]], p) * 1000) for p in (50, 95, 99)},
        "run_s": {p: round(pct([r["total"] for r in ok], p), 2) for p in (50, 95, 99)},
        "reject_ms": {p: round(pct([r["total"] for r in rejected], p) * 1000) for p in (50, 95)} if rejected else None,
        "health_ms": {
            "p50": round(pct([h for h, _ in health], 50) * 1000, 1),
            "max": round(max((h for h, _ in health), default=0) * 1000, 1),
            "peak_runs_in_flight": max((n or 0 for _, n in health), default=0),
        },
    }
    print(json.dumps(summary, indent=2))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
