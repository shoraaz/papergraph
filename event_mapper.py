"""
event_mapper.py — Agno Event to UI Step Translator
===================================================

Translates Agno's real run events (TeamRunContent, ToolCallCompleted, RunCompleted)
into structured UI step objects (plan, retrieve, grade, correct, synthesize, error).

Supports real-time token streaming via answer_deltas and formats step details.
For full details on event mapping rules and Agno event structures,
see `docs/DESIGN_DECISIONS.md`.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Cosmetic constants — must match the color/glyph vocabulary already
# defined in frontend/index.html's CSS custom properties.
# ---------------------------------------------------------------------------

COLORS = {
    "plan": "var(--blue)",
    "retrieve_vector": "var(--green)",
    "retrieve_graph": "var(--blue)",
    "grade_sufficient": "var(--green)",
    "grade_insufficient": "var(--yellow)",
    "correct": "var(--red)",
    "synthesize": "var(--accent)",
    "error": "var(--red)",
}

GLYPHS = {
    "plan": "◆",
    "retrieve_vector": "▤",
    "retrieve_graph": "⬡",
    "grade_sufficient": "✓",
    "grade_insufficient": "⊘",
    "correct": "↻",
    "synthesize": "◈",
    "error": "!",
}

# Retrieval tools, by exact name (tools.py owns the list). Anything else an
# agent might call is not part of the RAG story and isn't shown.
from tools import GRAPH_TOOLS, VECTOR_TOOLS  # noqa: E402

@dataclass
class TraceStep:
    kind: str
    glyph: str
    color: str
    title: str
    body: str
    ms: str = ""
    query: Optional[str] = None
    score: Optional[float] = None
    verdict: Optional[str] = None
    detail_label: str = ""
    detail: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "kind": self.kind,
            "glyph": self.glyph,
            "color": self.color,
            "title": self.title,
            "body": self.body,
            "ms": self.ms,
            "detailLabel": self.detail_label,
            "detail": self.detail,
        }
        if self.query is not None:
            d["query"] = self.query
        if self.score is not None:
            d["score"] = self.score
        if self.verdict is not None:
            d["verdict"] = self.verdict
        return d


def _event_name(event: Any) -> str:
    """The event's string discriminator if present, else its class name.
    Both have been observed in practice; check the string field first
    since it's the more stable/intentional API surface."""
    return getattr(event, "event", None) or type(event).__name__


class RunTraceBuilder:
    """
    Consumes Agno's live event stream for one team.run(...) call and
    incrementally yields TraceStep objects as they complete, plus the
    final answer text and any ClaimCheck objects encountered (routed
    separately since claims render as cards, not trace steps).

    Usage:
        builder = RunTraceBuilder()
        async for event in team.arun(message, stream=True, stream_events=True):
            steps, deltas = builder.feed(event)
            for step in steps:
                yield {"event": "step", "data": step.to_dict()}
            for delta in deltas:
                yield {"event": "delta", "data": {"text": delta}}
        final_answer = builder.final_answer
        claims = builder.claims
    """

    def __init__(self):
        self.final_answer: str = ""
        self.claims: list[dict] = []
        self.grounding_checks: list[dict] = []
        self.errored: bool = False
        self.error_message: str = ""
        self._pending_starts: dict[str, float] = {}
        self._last_grade_confidence: Optional[float] = None
        self._pending_correction: bool = False
        self._answer_buffer: list[str] = []

    def feed(self, event: Any) -> tuple[list[TraceStep], list[str]]:
        """
        Process one raw Agno event. Returns (steps, answer_deltas):
          steps: zero or more completed TraceSteps for the reasoning trace
          answer_deltas: zero or more text chunks of the LEADER's final
            answer, streamed as they arrive (see module docstring #3)
        """
        steps: list[TraceStep] = []
        deltas: list[str] = []
        name = _event_name(event)

        # --- Hard failure: surface once, stop processing further events ----
        if name in ("RunError", "TeamRunError", "TeamToolCallError", "ToolCallError"):
            if not self.errored:
                self.errored = True
                self.error_message = _extract_error_text(event)
                steps.append(
                    TraceStep(
                        kind="error",
                        glyph=GLYPHS["error"],
                        color=COLORS["error"],
                        title="Provider error",
                        body=self.error_message,
                        detail_label="RAW ERROR",
                        detail=[self.error_message],
                    )
                )
            return steps, deltas

        # --- Team-level delegation = a 'plan' step --------------------------
        if name == "TeamToolCallStarted":
            tool = getattr(event, "tool", None)
            args = getattr(tool, "tool_args", {}) if tool else {}
            tool_name = getattr(tool, "tool_name", "") if tool else ""
            if tool_name != "delegate_task_to_member":
                return steps, deltas
            member = args.get("member_id", "agent")
            task = args.get("task", "")

            if self._pending_correction:
                self._pending_correction = False
                steps.append(
                    TraceStep(
                        kind="correct",
                        glyph=GLYPHS["correct"],
                        color=COLORS["correct"],
                        title="Corrective RAG triggered — re-querying",
                        body=(
                            f"Grader confidence ({self._last_grade_confidence:.2f}) fell below "
                            "the 0.5 floor. Re-querying before generating an answer."
                        ),
                        detail_label="TRIGGER",
                        detail=[f"grader_confidence={self._last_grade_confidence:.2f} < 0.5 threshold"],
                    )
                )

            steps.append(
                TraceStep(
                    kind="plan",
                    glyph=GLYPHS["plan"],
                    color=COLORS["plan"],
                    title=f"Team leader delegates to {member}",
                    body=task,
                    detail_label="DELEGATION",
                    detail=[f'delegate_task_to_member(member_id="{member}", task="{task}")'],
                )
            )
            return steps, deltas

        # --- Member-level tool call started: remember start time -----------
        if name == "ToolCallStarted":
            tool = getattr(event, "tool", None)
            call_id = getattr(tool, "tool_call_id", None) if tool else None
            if call_id:
                self._pending_starts[call_id] = time.time()
            return steps, deltas

        # --- Member-level tool call completed: emit a 'retrieve' step -------
        if name == "ToolCallCompleted":
            tool = getattr(event, "tool", None)
            if not tool:
                return steps, deltas
            tool_name = getattr(tool, "tool_name", "") or ""
            args = getattr(tool, "tool_args", {}) or {}
            result = getattr(tool, "result", None)
            agent_name = getattr(event, "agent_name", "") or "Agent"
            call_id = getattr(tool, "tool_call_id", None)

            started_at = self._pending_starts.pop(call_id, None)
            elapsed_ms = f"{time.time() - started_at:.2f}s" if started_at else ""

            is_graph_tool = tool_name in GRAPH_TOOLS
            is_vector_tool = tool_name in VECTOR_TOOLS
            if not (is_graph_tool or is_vector_tool):
                return steps, deltas

            kind_key = "retrieve_graph" if is_graph_tool else "retrieve_vector"
            query_str = args.get("query") or args.get("paper_id") or args.get("concept") or ""
            filters = {k: v for k, v in args.items() if k in ("paper_id", "section") and v and is_vector_tool}
            if filters:
                query_str += "  [" + ", ".join(f"{k}={v}" for k, v in filters.items()) + "]"
            parsed = _parse_tool_result(result)
            result_rows = _rows_from_tool(tool_name, parsed)

            steps.append(
                TraceStep(
                    kind="retrieve",
                    glyph=GLYPHS[kind_key],
                    color=COLORS[kind_key],
                    title=f"{'Knowledge graph' if is_graph_tool else 'Vector store'} · {tool_name} · {agent_name}",
                    body=_summarize_retrieval(tool_name, parsed, result_rows),
                    ms=elapsed_ms,
                    query=query_str,
                    detail_label="GRAPH RESULT" if is_graph_tool else "RETRIEVED CHUNKS",
                    detail=_format_result_preview(result_rows),
                )
            )
            return steps, deltas

        # --- A member's RunCompleted: check for structured Grader/Claim output --
        if name == "RunCompleted":
            agent_name = getattr(event, "agent_name", "") or ""
            content = getattr(event, "content", None)

            if agent_name == "Grader" and content is not None and hasattr(content, "confidence"):
                score = float(content.confidence)
                sufficient = bool(content.is_sufficient)
                kind_key = "grade_sufficient" if sufficient else "grade_insufficient"
                verdict = "sufficient" if sufficient else "insufficient"

                self._last_grade_confidence = score
                self._pending_correction = score < 0.5

                steps.append(
                    TraceStep(
                        kind="grade",
                        glyph=GLYPHS[kind_key],
                        color=COLORS[kind_key],
                        title=f"Self-RAG grader · {verdict}",
                        body=content.reasoning,
                        score=score,
                        verdict=verdict,
                        detail_label="GRADER OUTPUT",
                        detail=[content.model_dump_json(indent=2)],
                    )
                )
                self.grounding_checks.append(content.model_dump())
                return steps, deltas

            if agent_name == "ClaimVerifier" and content is not None and hasattr(content, "claim"):
                self.claims.append(content.model_dump())
                return steps, deltas

        # --- Leader's final answer streaming in, token by token -------------
        # agent_name is None/empty specifically for the TEAM LEADER's own
        # content — a member's RunContent (agent_name set) is that member's
        # internal reasoning/response, not the final answer, and must not
        # be forwarded here (it would make the trace show fragments of
        # sub-agent chatter as if it were the synthesized answer).
        if name == "TeamRunContent" and getattr(event, "agent_name", None) in (None, ""):
            chunk = getattr(event, "content", None)
            if isinstance(chunk, str) and chunk:
                self._answer_buffer.append(chunk)
                deltas.append(chunk)
            return steps, deltas

        # --- Team's final synthesized answer (complete/accumulated) ---------
        if name == "TeamRunCompleted" and getattr(event, "agent_name", None) in (None, ""):
            content = getattr(event, "content", None)
            if isinstance(content, str) and content:
                self.final_answer = content
            elif self._answer_buffer:
                # Fallback: some providers/paths may not populate the final
                # TeamRunCompleted.content even though deltas streamed fine —
                # reconstruct from what we already forwarded rather than lose it.
                self.final_answer = "".join(self._answer_buffer)

        return steps, deltas


def _parse_tool_result(result: Any) -> Any:
    """tools.py returns JSON strings; Agno may hand them back as str or
    already-decoded. Normalize to Python objects."""
    if isinstance(result, str):
        import json

        try:
            return json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return result
    return result


def _rows_from_tool(tool_name: str, parsed: Any) -> list:
    if not isinstance(parsed, dict):
        return [] if parsed in (None, "") else [parsed]
    if "results" in parsed:  # search_corpus
        return [
            f"[{r.get('score')}] {r.get('paper_id')} · {r.get('section')} — {str(r.get('text', ''))[:160]}"
            for r in parsed["results"]
        ]
    if tool_name == "get_prerequisites":
        if not parsed.get("paper_found"):
            return []
        rows = [f"{s['step']}. {s['title']} ({s['paper_id']}) — {', '.join(s['covers'])}" for s in parsed.get("reading_order", [])]
        rows += [f"untaught: {c}" for c in parsed.get("untaught_concepts", [])]
        return rows or [f"outcome: {parsed.get('outcome')}"]
    if tool_name == "get_paper_details":
        if not parsed.get("paper_found"):
            return []
        return [
            f"{parsed.get('paper_id')} · {parsed.get('title')}",
            "teaches: " + ", ".join(parsed.get("teaches", [])),
            "requires: " + ", ".join(parsed.get("requires", [])),
        ]
    if "papers" in parsed:  # find_papers / papers_teaching_concept
        return [f"{p.get('paper_id')} · {p.get('title')}" for p in parsed["papers"]]
    if "error" in parsed:
        return [f"error: {parsed['error']}"]
    return [parsed]


def _format_result_preview(rows: list, max_items: int = 6) -> list[str]:
    if not rows:
        return ["(nothing returned)"]
    lines = [str(row)[:240] for row in rows[:max_items]]
    if len(rows) > max_items:
        lines.append(f"… {len(rows) - max_items} more")
    return lines


def _summarize_retrieval(tool_name: str, parsed: Any, rows: list) -> str:
    if tool_name == "get_prerequisites" and isinstance(parsed, dict):
        if not parsed.get("paper_found"):
            return "Paper not found in the corpus graph."
        outcome = parsed.get("outcome")
        n = max(len(parsed.get("reading_order", [])) - 1, 0)
        if outcome == "partial":
            return f"Reading plan with {n} prerequisite paper(s); {len(parsed.get('untaught_concepts', []))} concept(s) have no in-corpus teacher."
        if outcome == "no_prerequisites_recorded":
            return "Paper found; the graph records no prerequisites for it."
        return f"Complete reading plan: {n} prerequisite paper(s) before the target."
    if tool_name == "get_paper_details" and isinstance(parsed, dict):
        return "Paper found in the graph." if parsed.get("paper_found") else "Paper not found in the corpus graph."
    if not rows:
        return "Nothing relevant found."
    if tool_name in VECTOR_TOOLS:
        return f"Retrieved {len(rows)} chunk(s) from the corpus (hybrid dense + BM25)."
    return f"Graph lookup returned {len(rows)} result(s)."


def _extract_error_text(event: Any) -> str:
    for attr in ("content", "message", "error"):
        val = getattr(event, attr, None)
        if val:
            return str(val)[:500]
    return "Unknown provider error."
