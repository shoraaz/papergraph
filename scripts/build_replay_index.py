"""
Rebuild frontend/replays/index.json from the recorded runs in that folder.

Record real runs with your own key, locally:

    RECORD_RUNS_DIR=frontend/replays REQUIRE_BYOK=true uvicorn server:app --port 8000
    # ask 2-3 good questions in the UI (e.g. the DuetRAG claim check)

then run `python scripts/build_replay_index.py` and commit the JSON files.
Visitors without a key can watch those runs, played back from the CDN
with no backend call. Recordings contain the question, the trace and the
answer — never the key, session id or client address.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPLAYS = ROOT / "frontend" / "replays"


def main() -> int:
    entries = []
    for path in sorted(REPLAYS.glob("*.json")):
        if path.name == "index.json":
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        events = rec.get("events", [])
        if not events or events[-1].get("event") != "done" or events[-1]["data"].get("errored"):
            print(f"skip {path.name}: not a successful run")
            continue
        question = rec.get("question", "")
        entries.append(
            {
                "file": path.name,
                "question": question,
                "title": rec.get("title") or (question if len(question) <= 60 else question[:57] + "…"),
                "model": rec.get("model"),
                "recorded_at": rec.get("recorded_at"),
            }
        )
    (REPLAYS / "index.json").write_text(json.dumps({"replays": entries}, indent=2) + "\n", encoding="utf-8")
    print(f"{len(entries)} replay(s) indexed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
