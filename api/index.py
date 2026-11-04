"""
Vercel entrypoint. vercel.json rewrites /api/* here; the static frontend
is served by Vercel's CDN from frontend/.

Required project env vars on Vercel: none (defaults are the demo profile).
Recommended: UPSTASH_REDIS_REST_URL + UPSTASH_REDIS_REST_TOKEN so rate
limits are shared across instances, and CORS_ORIGINS=<your domain>.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import app  # noqa: E402,F401
