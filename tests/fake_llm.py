"""
A scripted, OpenAI-compatible chat-completions endpoint for end-to-end
tests of the REAL Agno team: real delegation, real tool execution against
the stores, real event stream, real event_mapper — only the model's
decisions are scripted. No network, no API key.

Each agent is recognised from what Agno sends it (its tools and system
prompt) and replies according to how many tool results it has already
seen in its own message history.
"""

from __future__ import annotations

import json
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

GRADE = {"is_sufficient": True, "confidence": 0.85, "reasoning": "Retrieved Self-RAG results passages.", "missing_info": None}
CLAIM = {
    "claim": "Self-RAG outperforms ChatGPT on open-domain QA.",
    "is_supported": True,
    "contradiction_found": False,
    "overstated": True,
    "scope_caveat": "Fixture corpus: one results chunk only.",
    "explanation": "The results chunk reports the comparison on a few benchmarks.",
}
FINAL = "Self-RAG emits self-reflection tokens to decide when to retrieve. The outperformance claim is overstated."


def _tool_names(body: dict) -> set[str]:
    return {t["function"]["name"] for t in body.get("tools") or []}


def _tool_results(body: dict) -> int:
    return sum(1 for m in body["messages"] if m.get("role") == "tool")


def _system(body: dict) -> str:
    return " ".join(str(m.get("content")) for m in body["messages"] if m.get("role") == "system")


def _member_ids(body: dict) -> dict[str, str]:
    """Read member ids exactly as the leader is shown them in its prompt —
    the same information a real model has."""
    import re

    return {name.lower(): mid for mid, name in re.findall(r'member id="([^"]+)" name="([^"]+)"', _system(body))}


def decide(body: dict) -> dict:
    """-> {"tool": (name, args)} or {"text": str}"""
    tools = _tool_names(body)
    seen = _tool_results(body)
    if "delegate_task_to_member" in tools:  # team leader
        member_ids = _member_ids(body)
        plan = [
            ("retriever", "Find results passages about Self-RAG vs ChatGPT."),
            ("grader", "Grade the retrieved context."),
            ("claimverifier", "Verify: Self-RAG outperforms ChatGPT on open-domain QA."),
        ]
        if seen < len(plan):
            member, task = plan[seen]
            return {"tool": ("delegate_task_to_member", {"member_id": member_ids[member], "task": task})}
        return {"text": FINAL}
    if "get_prerequisites" in tools and "search_corpus" in tools:  # Retriever
        if seen == 0:
            return {"tool": ("search_corpus", {"query": "Self-RAG outperforms ChatGPT results"})}
        return {"text": "search_corpus returned Self-RAG results passages (2310.11511v1)."}
    if "search_corpus" in tools:  # ClaimVerifier
        if seen == 0:
            return {"tool": ("search_corpus", {"query": "outperforms ChatGPT", "paper_id": "2310.11511", "section": "results"})}
        return {"text": json.dumps(CLAIM)}
    if "Grade retrieved context" in _system(body):
        return {"text": json.dumps(GRADE)}
    return {"text": "ok"}


def _chunk(cid: str, delta: dict, finish=None) -> str:
    payload = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "fake",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def make_app() -> FastAPI:
    app = FastAPI()
    app.state.requests = []

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        body = await request.json()
        app.state.requests.append(body)
        action = decide(body)
        cid = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

        if not body.get("stream"):
            msg = {"role": "assistant", "content": None}
            finish = "stop"
            if "tool" in action:
                name, args = action["tool"]
                msg["tool_calls"] = [
                    {"id": f"call_{uuid.uuid4().hex[:6]}", "type": "function",
                     "function": {"name": name, "arguments": json.dumps(args)}}
                ]
                finish = "tool_calls"
            else:
                msg["content"] = action["text"]
            return JSONResponse({"id": cid, "object": "chat.completion", "created": 0, "model": "fake",
                                 "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage})

        def gen():
            yield _chunk(cid, {"role": "assistant", "content": ""})
            if "tool" in action:
                name, args = action["tool"]
                yield _chunk(cid, {"tool_calls": [{"index": 0, "id": f"call_{uuid.uuid4().hex[:6]}", "type": "function",
                                                   "function": {"name": name, "arguments": json.dumps(args)}}]})
                yield _chunk(cid, {}, "tool_calls")
            else:
                text = action["text"]
                for i in range(0, len(text), 24):
                    yield _chunk(cid, {"content": text[i : i + 24]})
                yield _chunk(cid, {}, "stop")
            if (body.get("stream_options") or {}).get("include_usage"):
                yield "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": 0,
                                             "model": "fake", "choices": [], "usage": usage}) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app
