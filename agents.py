"""
agents.py — PaperGraph Agent Team Architecture
===============================================

Orchestrates a 6-agent team using Agno to implement Self-RAG and Corrective RAG:
  - RetrieverAgent     : Decides vector (Qdrant) vs graph (Neo4j) retrieval.
  - GraderAgent        : Self-RAG grounding check (returns GroundingCheck schema).
  - SequencerAgent     : GraphRAG Cypher traversal for prerequisite reading order.
  - JargonAgent        : Simplifies dense technical sections into plain English.
  - ClaimVerifierAgent : Corrective RAG verifier auditing abstract claims against results (ClaimCheck).
  - PaperGraphTeam     : Leader orchestrating team delegation, memory (Mem0), and final synthesis.

For full technical rationale, failure-mode mitigations, and schema details,
see `docs/DESIGN_DECISIONS.md`.
"""

import os
from dataclasses import dataclass
from typing import Optional

from agno.agent import Agent
from agno.models.base import Model
from agno.team.team import Team
from agno.tools.knowledge import KnowledgeTools
from agno.tools.mem0 import Mem0Tools
from agno.tools.neo4j import Neo4jTools
from pydantic import BaseModel, Field

from db import (
    EMBEDDER_DIMENSIONS,
    EMBEDDER_MODEL,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    QDRANT_API_KEY,
    QDRANT_URL,
    SHARED_WORKSPACE,
    get_knowledge,
)

# ---------------------------------------------------------------------------
# Provider-aware model selection (BYOK)
# ---------------------------------------------------------------------------

DEFAULT_PROVIDER = "gemini"
DEFAULT_GEMINI_MODEL = os.environ.get("PAPERGRAPH_MODEL", "gemini-2.0-flash")

PROVIDER_DEFAULT_MODEL = {
    "gemini": DEFAULT_GEMINI_MODEL,
    "groq": "llama-3.3-70b-versatile",
    "openrouter": "meta-llama/llama-3.1-8b-instruct:free",
}


@dataclass
class ModelChoice:
    """
    Which provider/model/key to run the whole Team on for one request.

    provider: "gemini" | "groq" | "openrouter"
    model_id: provider-specific model identifier (e.g. "openai/gpt-oss-120b")
    api_key:  the caller's own key. Never persisted — held only for the
              lifetime of the ModelChoice object / the request that built it.
    """
    provider: str = DEFAULT_PROVIDER
    model_id: Optional[str] = None
    api_key: Optional[str] = None

    def resolved_model_id(self) -> str:
        return self.model_id or PROVIDER_DEFAULT_MODEL.get(self.provider, DEFAULT_GEMINI_MODEL)


def _build_model(choice: Optional[ModelChoice] = None) -> Model:
    """
    Construct the actual Agno model object for one ModelChoice. Each
    provider's Agno class takes api_key at construction time (not just
    via env var), which is exactly what BYOK needs — a key that's used
    once, per-request, and never written anywhere.
    """
    choice = choice or ModelChoice()
    model_id = choice.resolved_model_id()

    if choice.provider == "groq":
        from agno.models.groq import Groq
        return Groq(id=model_id, api_key=choice.api_key or os.environ.get("GROQ_API_KEY"))

    if choice.provider == "openrouter":
        from agno.models.openrouter import OpenRouter
        return OpenRouter(id=model_id, api_key=choice.api_key or os.environ.get("OPENROUTER_API_KEY"))

    # Default / "gemini": plain Google AI Studio API (not Vertex) — no
    # cloud console dependency, identical wherever this is hosted.
    from agno.models.google import Gemini
    return Gemini(id=model_id, api_key=choice.api_key or os.environ.get("GOOGLE_API_KEY"))


def _neo4j_tools() -> Neo4jTools:
    """
    Neo4jTools falls back to NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD env
    vars automatically, but NOT to a NEO4J_DATABASE env var — AuraDB
    instances are scoped to a specific database name (not always the
    default 'neo4j'), so it must be passed explicitly here.
    """
    return Neo4jTools(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
    )


def _graph_schema_description(workspace_id: Optional[str]) -> str:
    """
    The real Neo4j schema, spelled out once here so every agent's
    instructions can reference the same ground truth instead of each
    re-describing (and potentially mis-describing) it independently.
    Keep this in sync with db.py's SCHEMA_CYPHER / node-shape comment.

    workspace_id: if given, adds an explicit instruction that every
    Cypher query MUST filter Paper nodes to workspace_id IN
    [$workspace_id, 'shared'] — this is the Neo4j half of workspace
    isolation (Qdrant's half is enforced server-side, see db.py).
    """
    workspace_clause = ""
    if workspace_id:
        workspace_clause = f"""

WORKSPACE ISOLATION — MANDATORY on every query against :Paper nodes:
This session's workspace_id is '{workspace_id}'. Every Cypher query that
matches (:Paper) must include a WHERE clause restricting to:
  p.workspace_id IN ['{workspace_id}', '{SHARED_WORKSPACE}']
This is required even when the user's question doesn't mention workspaces
at all — it silently scopes results to the shared base corpus plus this
session's own uploaded papers, never another session's uploads. Concept
nodes are NOT workspace-scoped (shared across everyone) so no filter is
needed on (:Concept) matches themselves.
"""

    return f"""
Neo4j graph schema (exact property names — use these, do not guess):
  (:Paper {{paper_id, title, authors, abstract, published_date, arxiv_url, workspace_id}})
  (:Concept {{name, display_name}})
  (:Paper)-[:CITES]->(:Paper)
  (:Paper)-[:TEACHES]->(:Concept)   -- this paper is a good source to LEARN this concept from
  (:Paper)-[:REQUIRES]->(:Concept)  -- this paper ASSUMES the reader already knows this concept

IMPORTANT: paper_id includes the arXiv version suffix (e.g. '2502.01113v3'),
but users will often give you an ID WITHOUT the version (e.g. '2502.01113').
Never match paper_id with exact equality on a user-provided ID — always use
a prefix match, e.g.:
  MATCH (p:Paper) WHERE p.paper_id STARTS WITH '2502.01113' RETURN p

CRITICAL — you MUST explicitly state what that STARTS WITH query actually
returned before concluding anything. Do not run the query and then reason
about absence from memory or assumption. If the tool result contains one
or more rows, THE PAPER IS IN THE CORPUS — quote the returned title back
to yourself as confirmation, then proceed with the actual question. Only
say "not in the corpus" if the STARTS WITH query's result was a literal
empty list. Running the right query is not enough — you must also
correctly read what it gave back.
{workspace_clause}""".strip()


# ---------------------------------------------------------------------------
# Structured outputs for the two checker agents.
# These are what make Self-RAG / Corrective-RAG visible in the UI as
# real data, not something scraped out of prose.
# ---------------------------------------------------------------------------

class GroundingCheck(BaseModel):
    """Self-RAG grading of a retrieval result before the team answers."""
    is_sufficient: bool = Field(
        description="True if the retrieved context is relevant AND sufficient to answer the query."
    )
    confidence: float = Field(
        description="0.0-1.0. Below ~0.5 should trigger a refusal or a broader re-query."
    )
    reasoning: str = Field(
        description="One or two sentences on why the retrieval was/wasn't sufficient."
    )
    missing_info: str | None = Field(
        default=None,
        description="If insufficient, what specifically is missing from the retrieved context.",
    )


class ClaimCheck(BaseModel):
    """Corrective RAG: does a claim in the abstract/intro hold up against
    the paper's own results/experiments section?

    Three-way outcome, not just true/false — a claim can be numerically
    accurate (is_supported=True, contradiction_found=False) while still
    being scope-overstated (e.g. a result on one dataset/one base model
    presented in the abstract as a general improvement). overstated is
    a distinct signal from contradiction_found: an overstated claim
    isn't false, it's broader than its own evidence.
    """
    claim: str = Field(description="The claim being checked, restated concisely.")
    is_supported: bool = Field(
        description="True if the results/experiments section backs this claim as stated."
    )
    contradiction_found: bool = Field(
        description="True if the results section actively contradicts the claim (not just narrower in scope)."
    )
    overstated: bool = Field(
        default=False,
        description=(
            "True if the claim is technically accurate per the results, but the paper's abstract/intro "
            "frames it more broadly or generally than the actual experimental scope supports."
        ),
    )
    scope_caveat: str | None = Field(
        default=None,
        description=(
            "If overstated=True, the specific limitation the abstract glosses over — e.g. 'evaluated "
            "only on HotPotQA with a single base model (LLaMA2-7B); no cross-dataset or cross-model "
            "validation.' Null if overstated=False."
        ),
    )
    explanation: str = Field(
        description="What the results section actually shows, in plain terms."
    )


# ---------------------------------------------------------------------------
# Sub-agents
# ---------------------------------------------------------------------------

def build_retriever_agent(
    model_choice: Optional[ModelChoice] = None,
    workspace_id: Optional[str] = None,
) -> Agent:
    """Decides retrieval strategy per query: vector search (Qdrant, via
    KnowledgeTools' think->search->analyze loop) and/or graph traversal
    (Neo4j, via Neo4jTools' Cypher access) — this pairing is the
    'agentic RAG' mechanism: the agent chooses its own tools per query."""
    knowledge = get_knowledge(workspace_id=workspace_id)
    return Agent(
        name="Retriever",
        role="Retrieve relevant chunks and/or graph paths for a query, choosing the right tool(s).",
        model=_build_model(model_choice),
        tools=[
            KnowledgeTools(knowledge=knowledge, enable_think=True, enable_search=True, enable_analyze=True),
            _neo4j_tools(),
        ],
        instructions=[
            "Use KnowledgeTools for semantic/content questions (jargon, claims, 'what does this paper say').",
            "Use Neo4jTools for structural questions (prerequisites, citations, 'what should I read first').",
            "Use both when a query needs content grounded in graph structure.",
            "Your job is ONLY to retrieve and report what you found (or didn't find) — do not decide "
            "whether it's sufficient to answer the question. That decision belongs to the Grader, "
            "downstream. Report your findings plainly, including an honest 'nothing relevant found' "
            "when that's the case, and stop there.",
            _graph_schema_description(workspace_id),
        ],
    )


def build_grader_agent(model_choice: Optional[ModelChoice] = None) -> Agent:
    """Self-RAG checker: grades whatever the Retriever found before the
    team is allowed to answer from it."""
    return Agent(
        name="Grader",
        role="Grade retrieved context for relevance and sufficiency before an answer is generated.",
        model=_build_model(model_choice),
        output_schema=GroundingCheck,
        instructions=[
            "Be strict. If the retrieved context does not actually address the question, is_sufficient=False.",
            "Low confidence (<0.5) should be common when the corpus genuinely lacks the paper/concept asked about.",
            "Never let is_sufficient=True paper over a gap just because *something* was retrieved.",
            "An empty graph or empty vector search result is a textbook case of is_sufficient=False, "
            "confidence near 0.0 — not a prompt to reason from general knowledge instead.",
            "Grade the ACTUAL content the Retriever found (visible in the member interaction context "
            "you're given), not just the task description you were handed. If the Retriever's findings "
            "include real, relevant paragraphs of paper content, that is NOT an empty/insufficient case "
            "— check what was actually retrieved before concluding is_sufficient=False.",
            "Even if the Retriever explicitly reported 'nothing found', you still produce a formal "
            "grade for it (is_sufficient=False, confidence near 0.0, reasoning explaining the gap) "
            "rather than being skipped — your structured grade is what the UI shows the user.",
        ],
    )


def build_sequencer_agent(
    model_choice: Optional[ModelChoice] = None,
    workspace_id: Optional[str] = None,
) -> Agent:
    """GraphRAG: prerequisite reading-order sequencing via Neo4j traversal.
    The actual topological sort lives in graph.py; this agent calls it
    as a tool and explains the result."""
    return Agent(
        name="Sequencer",
        role="Determine prerequisite reading order for a paper using the concept-dependency graph.",
        model=_build_model(model_choice),
        tools=[_neo4j_tools()],
        instructions=[
            _graph_schema_description(workspace_id),
            "Query REQUIRES edges from the target paper to find prerequisite concepts.",
            "For each prerequisite concept, find papers that TEACH it (not just require it).",
            "Present the reading order as a sequence with a one-line reason for each step.",
            "CASE 1 — paper genuinely not in corpus: only conclude this after a STARTS WITH prefix "
            "match on paper_id genuinely returns an EMPTY result — and you must say what the query "
            "returned before concluding absence (per the schema note above). If confirmed empty, or a "
            "prerequisite query errors out, your entire response must be a short, plain refusal — e.g. "
            "'This paper isn't in my ingested corpus yet, so I can't determine prerequisites from the "
            "graph.' Stop there. Do NOT follow that sentence with a reading list assembled from your "
            "own general knowledge of the field, even as a 'here's what I'd suggest instead' addendum.",
            "CASE 2 — a DIFFERENT case, do not conflate it with Case 1: the target paper IS found and "
            "its prerequisite concepts ARE found, but no in-corpus paper has a TEACHES edge to any of "
            "those concepts. This is NOT 'paper not in corpus' — do not use the Case 1 refusal here. "
            "Instead report what you genuinely found: name the target paper, list the prerequisite "
            "concepts the graph identified, and say plainly the current corpus doesn't include a paper "
            "that teaches them, so you can't recommend a specific in-corpus reading order for those "
            "prerequisites. This is a real, useful partial answer, not a failure — state what worked "
            "(paper found, prerequisites identified) as clearly as what didn't (no teaching source "
            "available yet). You MAY suggest well-known external (non-corpus) papers or resources for "
            "each prerequisite concept if you're confident they're genuinely foundational works — but "
            "any such suggestion MUST be clearly and explicitly labeled as coming from your own general "
            "knowledge, NOT from the corpus (e.g. a distinct 'suggested external reading (not in "
            "corpus)' section). Never blend an external suggestion into the graph-grounded findings as "
            "if it came from the same source — the user must always be able to tell corpus-grounded "
            "facts apart from your own general knowledge at a glance.",
            "A graph-shaped answer must come from the graph. Case 1 is a hard refusal; Case 2 is an "
            "honest partial answer that still reports genuine findings — do not compress Case 2 down "
            "into Case 1's terse refusal, that discards real, correct work the graph query already did.",
        ],
    )


def build_jargon_agent(model_choice: Optional[ModelChoice] = None) -> Agent:
    """Paragraph-by-paragraph plain-English rewrites of dense sections."""
    return Agent(
        name="JargonDecoder",
        role="Rewrite dense academic paragraphs in plain English, one paragraph at a time.",
        model=_build_model(model_choice),
        instructions=[
            "Preserve technical accuracy — simplify language, not meaning.",
            "Keep the original paragraph boundaries so the rewrite maps 1:1 to the source.",
            "Define any term you can't avoid using, briefly, inline.",
            "Work from the actual retrieved paper content visible in your member interaction context "
            "— rewrite that specific text, don't write a generic explanation of the topic from memory.",
        ],
    )


def build_claim_verifier_agent(
    model_choice: Optional[ModelChoice] = None,
    workspace_id: Optional[str] = None,
) -> Agent:
    """Corrective RAG checker: cross-checks abstract/intro claims against
    the paper's own results section. This is the agent whose output
    should visibly trigger a correction/re-query moment in the UI."""
    return Agent(
        name="ClaimVerifier",
        role="Check whether claims in a paper's abstract/intro are backed by its own results section.",
        model=_build_model(model_choice),
        tools=[KnowledgeTools(knowledge=get_knowledge(workspace_id=workspace_id), enable_search=True, enable_analyze=True)],
        output_schema=ClaimCheck,
        instructions=[
            "Search specifically within the results/experiments sections, not the abstract, for evidence.",
            "Check three distinct things, not just true/false:",
            "1. is_supported — do the numbers/results in the paper actually back the claim as measured?",
            "2. contradiction_found — does the results section actively contradict or undercut the claim?",
            "3. overstated — even if is_supported=True and contradiction_found=False, does the "
            "abstract/intro frame the claim more broadly or generally than the actual experimental "
            "scope supports? E.g. a result on ONE dataset and ONE base model, presented as a general "
            "improvement. This is common and worth catching — check what datasets, models, and "
            "conditions the results section actually covers vs. how the abstract phrases the claim.",
            "If overstated=True, fill scope_caveat with the specific limitation being glossed over.",
            "If the results section is not in the retrieved context, say so rather than guessing.",
        ],
    )


# ---------------------------------------------------------------------------
# Team
# ---------------------------------------------------------------------------

def build_papergraph_team(
    session_user_id: str,
    model_choice: Optional[ModelChoice] = None,
) -> Team:
    """
    session_user_id: an anonymous session UUID today; a real logged-in
    user_id later. Doubles as workspace_id for isolating this session's
    own uploaded papers (see module docstring's note on workspace
    isolation) — one identifier, one meaning: "this visitor's stuff."

    model_choice: which provider/model/key to run every agent AND the
    Team leader on for this request. None => Gemini via GOOGLE_API_KEY
    env var (today's default CLI behavior, unchanged).

    Mem0 itself always runs on Gemini/BGE/Qdrant regardless of
    model_choice — Mem0Tools takes its own MEM0_CONFIG below, separate
    from the user-facing agents, since memory is infrastructure the
    user's BYOK key shouldn't need to pay for.
    """
    workspace_id = session_user_id

    mem0_config = {
        "llm": {
            "provider": "gemini",
            "config": {
                "model": DEFAULT_GEMINI_MODEL,
                "api_key": os.environ.get("GOOGLE_API_KEY"),
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": EMBEDDER_MODEL,
                "embedding_dims": EMBEDDER_DIMENSIONS,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "papergraph_memory",
                "url": QDRANT_URL,
                "api_key": QDRANT_API_KEY,
                "embedding_model_dims": EMBEDDER_DIMENSIONS,
            },
        },
    }

    return Team(
        name="PaperGraph",
        model=_build_model(model_choice),
        members=[
            build_retriever_agent(model_choice, workspace_id=workspace_id),
            build_grader_agent(model_choice),
            build_sequencer_agent(model_choice, workspace_id=workspace_id),
            build_jargon_agent(model_choice),
            build_claim_verifier_agent(model_choice, workspace_id=workspace_id),
        ],
        tools=[Mem0Tools(config=mem0_config, user_id=session_user_id)],
        # Forwards prior members' verbatim responses into subsequent
        # members' task context — required so e.g. Grader actually sees
        # what Retriever found, not just the leader's paraphrased task.
        # See module docstring for the real bug this fixes.
        share_member_interactions=True,
        instructions=[
            "Plan which member(s) a query needs before delegating — state the plan briefly.",
            "MANDATORY for EVERY content/jargon/claim query, no exceptions: delegate to Retriever "
            "FIRST, then ALWAYS delegate to Grader next — even if Retriever's own response already "
            "reads like a confident 'nothing found' or a confident 'here's the answer'. Never treat "
            "Retriever's own framing as a substitute for Grader's formal grade. Skipping Grader because "
            "the answer already 'seems clear' is exactly the failure mode this rule exists to prevent — "
            "the Grader step must actually run and produce a GroundingCheck every single time, so its "
            "confidence score is something the UI can always show.",
            "If Grader returns is_sufficient=False or confidence<0.5, say so explicitly (state the "
            "confidence score) and either re-query more broadly once, or refuse honestly rather than "
            "answering from ungrounded context.",
            "If Grader returns is_sufficient=True, use the Retriever's actual retrieved content (not "
            "your own general knowledge) to answer — for jargon-decoding requests, delegate that "
            "retrieved content to JargonDecoder rather than re-answering yourself.",
            "For prerequisite/reading-order questions: delegate to Sequencer. If Sequencer's response "
            "reports genuine partial findings (e.g. 'paper found, prerequisites identified, but no "
            "in-corpus paper teaches them') rather than a flat 'not in corpus' refusal, PASS THAT "
            "NUANCE THROUGH — do not compress a genuine partial finding down into the same terse "
            "refusal used for 'paper not in corpus at all'. These are different outcomes and the user "
            "should be able to tell them apart from your final answer. If Sequencer included clearly-"
            "labeled external (non-corpus) suggestions, preserve that labeling distinction in your "
            "final answer too — never let it read as if those suggestions came from the corpus.",
            "For claim-checking questions: delegate to ClaimVerifier. Surface all three outcomes "
            "distinctly in your final answer: if contradiction_found=True, lead with that as a clear "
            "correction; if overstated=True, clearly flag the scope_caveat so the user knows the claim "
            "is narrower than it reads; don't collapse 'overstated' into either 'supported' or "
            "'contradicted' — it's its own category and should read as its own category to the user.",
            "HARD RULE, no exceptions: if a member's response says a paper/concept is GENUINELY not in "
            "the corpus (an empty query result, confirmed absence) — your final reply to the user MUST "
            "be that refusal, stated plainly, and NOTHING ELSE. Do not add a 'here's what I can tell "
            "you from general knowledge instead' section. Do not soften it with a fallback answer. Do "
            "not treat the member's refusal as a cue to answer the question yourself from your own "
            "training data — that is the exact failure mode this rule exists to prevent. The refusal "
            "IS the complete, correct answer in that case. This hard rule applies to genuine absence "
            "only — it does not apply to the partial-findings case described above, which should be "
            "reported with its actual nuance (including clearly-labeled external suggestions if "
            "present), not flattened into this same refusal shape.",
        ],
        show_members_responses=True,
        add_datetime_to_context=True,
        markdown=True,
    )


if __name__ == "__main__":
    import uuid

    team = build_papergraph_team(session_user_id=str(uuid.uuid4()))
    team.print_response(
        "What should I read before tackling a paper on retrieval-augmented generation?",
        stream=True,
    )
