"""
Vercel entrypoint. Vercel turns every .py file in api/ into a function;
vercel.json rewrites /api/* here, and frontend/ is served from the CDN.

Required project env vars on Vercel: none (defaults are the demo profile).
Recommended: UPSTASH_REDIS_REST_URL + UPSTASH_REDIS_REST_TOKEN so rate
limits are shared across instances, and CORS_ORIGINS=<your domain>.
Function duration: Project Settings -> Functions -> Max Duration 300 s
(long agent runs stream for minutes).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server  # noqa: E402

app = server.app
