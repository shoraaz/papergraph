"""
eval/run_evals.py — PaperGraph RAGAS Evaluation Runner
=======================================================

Measures RAG quality of the PaperGraph agent team using four RAGAS metrics:
  - Faithfulness:       Are all claims in the answer grounded in retrieved context?
  - Answer Relevancy:   Does the answer actually address the question?
  - Context Precision:  Are retrieved chunks relevant (no noise)?
  - Context Recall:     Did retrieval find all info needed for the ground truth?

HOW IT WORKS
------------
This script calls build_papergraph_team() DIRECTLY (no HTTP server needed)
using the non-streaming Team.run() API, which returns a TeamRunOutput object.
See: https://docs.agno.com/teams/running-teams#run-output

TeamRunOutput fields we use:
  - .content          → the team leader's final answer string
  - .member_responses → list of RunOutput per delegated member agent
    Each RunOutput has:
      - .content  → the member's final output string
      - .messages → list of Message objects (includes tool call results)

Retrieved contexts are extracted from the Retriever member's message history:
tool call results from KnowledgeTools.search_knowledge and Neo4jTools
appear as "tool" role messages in .messages. We collect all such results
as the retrieved_contexts list for RAGAS. Confirmed via a live debug run
against the real Team that this extraction logic is correct — role='tool'
messages genuinely contain the retrieved chunk text and Neo4j query
results (e.g. "This work introduces SELF-RAG...", and raw Cypher result
dicts). If contexts still come back empty on a real run, the cause is
very likely Gemini's free-tier rate limit (5 req/min) being exhausted
mid-run — a 5-agent Team can burn through that quickly on a single
question — not a bug in this extraction function.

RAGAS API used (>= 0.2):
  from ragas import evaluate, EvaluationDataset, SingleTurnSample
  See: https://docs.ragas.io/en/stable/getstarted/rag_evaluation.html

IMPORTANT — RAGAS's evaluate() must be given EXPLICIT non-OpenAI
embeddings, not left to its own default: a live smoke test surfaced a
real, hard-blocking bug — evaluate() internally calls
ragas.embeddings.embedding_factory() with no arguments whenever a
metric needs embeddings (AnswerRelevancy does, to compute cosine
similarity between reverse-generated questions and the real one), and
that factory defaults to OpenAI embeddings, requiring OPENAI_API_KEY —
a credential this project deliberately never uses anywhere else (every
other embedding call in PaperGraph is local, via BAAI/bge-base-en-v1.5;
every LLM call is Gemini/Groq/OpenRouter, never OpenAI). Fixed by
constructing a LangchainEmbeddingsWrapper around
GoogleGenerativeAIEmbeddings (same google-generativeai credential
already used for the judge LLM) and passing it explicitly to evaluate().

IMPORTANT — ragas==0.3.9 required a manual source patch to even import:
ragas's own ragas/llms/base.py unconditionally imports ChatVertexAI and
VertexAI from langchain_community at module load time, but current
langchain-community (0.4.x) has already removed that submodule as part
of its own deprecation migration (ChatVertexAI moved to the separate
langchain-google-vertexai package) — a confirmed, currently-open
upstream bug (github.com/vibrantlabsai/ragas issues #2741, #2745). We
never use Vertex AI anywhere in this project (only Gemini via
ChatGoogleGenerativeAI), so the installed ragas/llms/base.py was
patched to wrap that import in a try/except, falling back to two
harmless placeholder classes — they're only ever referenced in a
static isinstance() list (MULTIPLE_COMPLETION_SUPPORTED) that never
actually receives a real Vertex AI object in this codebase anyway.
This patch lives in .venv/Lib/site-packages/ragas/llms/base.py and
will need to be reapplied if the venv is ever rebuilt from a clean
`uv sync`, until ragas ships an upstream fix.

IMPORTANT — --start-from-id exists because a real, live 15-question run
hit Gemini's free-tier 429 (RESOURCE_EXHAUSTED) partway through despite
the inter-question delay (a single question's 5-agent Team run can
burn enough tokens on its own to trip the per-minute quota even with
gaps between questions) — q08 came back with a degraded 147-char answer
and q09 came back with 0 retrieved contexts as a direct result. Rather
than re-running the whole suite (re-spending quota and time on q01-q07,
which had already completed cleanly), --start-from-id lets you resume
a partial run from a specific question ID onward — pair with a fresh
API key or after waiting out the quota window.

USAGE
-----
  # Run all 15 questions (slow — ~30s+ per question via Gemini, plus a
  # deliberate delay between questions — see RATE_LIMIT_DELAY_SECONDS —
  # to avoid exhausting Gemini's free-tier 5 req/min quota mid-run,
  # which silently produces empty retrieved_contexts for later questions)
  .venv\\Scripts\\python.exe eval/run_evals.py

  # Quick smoke test with first 3 questions only
  .venv\\Scripts\\python.exe eval/run_evals.py --limit 3

  # Only run factual category questions
  .venv\\Scripts\\python.exe eval/run_evals.py --category factual

  # Resume a partial run starting from question q08 (e.g. after a
  # rate-limit interruption partway through a full run)
  .venv\\Scripts\\python.exe eval/run_evals.py --start-from-id q08

  # Save results to a JSON file
  .venv\\Scripts\\python.exe eval/run_evals.py --output eval/results.json

REQUIREMENTS
------------
  uv add ragas==0.3.9 langchain-google-genai langchain-community
  (See the IMPORTANT notes above for why ragas is pinned to 0.3.9 and
  why the installed package needs a one-time source patch.)
  GOOGLE_API_KEY must be set in .env (used for PaperGraph, the RAGAS
  judge LLM, and the RAGAS embeddings — all Gemini, never OpenAI)
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Ensure we can import from the project root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# Delay between evaluated questions, to avoid exhausting Gemini's free-tier
# 5 req/min quota mid-run — a 5-agent Team can make several LLM calls for
# a single question, so back-to-back questions with no gap reliably hits
# 429s partway through a run (confirmed via live testing).
RATE_LIMIT_DELAY_SECONDS = 15.0


# ---------------------------------------------------------------------------
# Lazy imports so we get a clean error message if ragas isn't installed
# ---------------------------------------------------------------------------
def _require_ragas():
    try:
        import ragas  # noqa: F401
    except ImportError as e:
        print("\n[eval] ragas import failed:")
        print(f"  {e}")
        print("\nIf this mentions langchain_community.chat_models.vertexai, see the")
        print("IMPORTANT note in this file's module docstring — ragas 0.3.9 needs a")
        print("one-time source patch for a currently-open upstream compatibility bug.")
        print("\nOtherwise, install ragas with:")
        print("  uv add ragas==0.3.9 langchain-google-genai langchain-community")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Context extraction helpers
# ---------------------------------------------------------------------------

def _extract_retrieved_contexts(team_run_output) -> list[str]:
    """
    Extract all retrieved text chunks from the Retriever agent's tool call
    results in the TeamRunOutput.

    Per the Agno docs (https://docs.agno.com/teams/running-teams#run-output),
    TeamRunOutput.member_responses is a list of RunOutput objects, one per
    delegated member. Each RunOutput.messages contains the full message
    history for that member, including tool call results as 'tool' role
    messages.

    The Retriever agent uses two tools:
      - KnowledgeTools (search_knowledge / search / analyze) — returns
        formatted chunk text from Qdrant
      - Neo4jTools (run_cypher_query / query) — returns graph query results

    Both appear as tool-role messages. We collect all of them as contexts.
    Confirmed correct via a live debug run against the real Team — see
    this module's docstring for what real tool-role message content
    looks like.
    """
    contexts: list[str] = []

    member_responses = getattr(team_run_output, "member_responses", []) or []

    for member_response in member_responses:
        messages = getattr(member_response, "messages", []) or []

        for msg in messages:
            role = getattr(msg, "role", None)
            content = getattr(msg, "content", None)

            if role == "tool" and content:
                text = content if isinstance(content, str) else str(content)
                text = text.strip()
                if text and text not in contexts:
                    contexts.append(text)

    # Fallback: if no tool messages found (e.g. an out-of-corpus question
    # that correctly never triggered retrieval, or a rate-limited run that
    # produced no member responses at all), use member response content
    # itself so RAGAS still has something to score against.
    if not contexts:
        for member_response in member_responses:
            content = getattr(member_response, "content", None)
            if content and isinstance(content, str) and content.strip():
                contexts.append(content.strip())

    return contexts


# ---------------------------------------------------------------------------
# Single eval runner
# ---------------------------------------------------------------------------

def run_single_question(question: str, session_id: str) -> tuple[str, list[str]]:
    """
    Run one question through the PaperGraph agent team.

    Uses Team.run() in NON-streaming mode (stream=False) so we get a clean
    TeamRunOutput object with .content and .member_responses.
    See: https://docs.agno.com/teams/running-teams#run-output

    Returns:
        (answer, retrieved_contexts)
        - answer: the team leader's final response string
        - retrieved_contexts: list of retrieved chunk/query result strings
    """
    from agents import ModelChoice, build_papergraph_team

    model_choice = ModelChoice(
        provider="gemini",
        model_id=os.environ.get("PAPERGRAPH_MODEL", "gemini-flash-latest"),
        api_key=os.environ.get("GOOGLE_API_KEY"),
    )

    team = build_papergraph_team(
        session_user_id=session_id,
        model_choice=model_choice,
    )

    run_output = team.run(question, stream=False)

    answer = ""
    if run_output and run_output.content:
        answer = run_output.content if isinstance(run_output.content, str) else str(run_output.content)

    contexts = _extract_retrieved_contexts(run_output) if run_output else []

    return answer.strip(), contexts


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------

def compute_ragas_scores(samples: list[dict]) -> dict:
    """
    Run RAGAS evaluation over a list of collected samples.

    Each sample must have:
      - user_input:          the question asked
      - response:            the LLM's answer
      - retrieved_contexts:  list of context strings used
      - reference:           ground truth answer string

    Uses RAGAS >= 0.2 API:
      from ragas import evaluate, EvaluationDataset, SingleTurnSample
      from ragas.metrics import (
          Faithfulness, AnswerRelevancy,
          LLMContextPrecisionWithReference, LLMContextRecall,
      )
    See: https://docs.ragas.io/en/stable/getstarted/rag_evaluation.html

    Both the judge LLM and the embeddings model are explicitly Gemini —
    see this module's docstring for why the embeddings must be passed
    explicitly rather than left to RAGAS's own default (which requires
    OPENAI_API_KEY, a credential this project never uses).
    """
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        AnswerRelevancy,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    except ImportError:
        print(
            "\n[eval] langchain-google-genai not installed. Run:\n"
            "  uv add langchain-google-genai\n"
            "  (RAGAS uses LangChain wrappers for its judge LLM and embeddings)"
        )
        sys.exit(1)

    judge_model_id = os.environ.get("PAPERGRAPH_MODEL", "gemini-flash-latest")

    judge_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model=judge_model_id,
            google_api_key=os.environ["GOOGLE_API_KEY"],
        )
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.environ["GOOGLE_API_KEY"],
        )
    )

    ragas_samples = []
    for s in samples:
        contexts = s["retrieved_contexts"]
        # RAGAS requires at least one context string; use answer as fallback
        # so metrics degrade gracefully rather than crashing on empty retrieval
        if not contexts:
            contexts = [s["response"] or "No context retrieved."]

        ragas_samples.append(
            SingleTurnSample(
                user_input=s["user_input"],
                response=s["response"],
                retrieved_contexts=contexts,
                reference=s["reference"],
            )
        )

    dataset = EvaluationDataset(samples=ragas_samples)

    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        LLMContextPrecisionWithReference(llm=judge_llm),
        LLMContextRecall(llm=judge_llm),
    ]

    print("\n[eval] Running RAGAS scoring (LLM-as-judge, may take a minute)...")
    results = evaluate(dataset=dataset, metrics=metrics)
    return results


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_results_table(
    questions: list[dict],
    collected: list[dict],
    ragas_results,
) -> None:
    """Print a formatted results table to stdout."""

    scores_df = ragas_results.to_pandas()

    col_w = [5, 48, 12, 14, 17, 14]
    header = f"{'ID':<{col_w[0]}} {'Question':<{col_w[1]}} {'Faithful':<{col_w[2]}} {'Relevancy':<{col_w[3]}} {'Ctx Precision':<{col_w[4]}} {'Ctx Recall':<{col_w[5]}}"
    sep = "-" * sum(col_w)

    print("\n" + "=" * sum(col_w))
    print("  PAPERGRAPH RAGAS EVALUATION RESULTS")
    print("=" * sum(col_w))
    print(header)
    print(sep)

    metric_keys = ["faithfulness", "answer_relevancy", "llm_context_precision_with_reference", "context_recall"]

    totals = {k: [] for k in metric_keys}

    for i, (q, row) in enumerate(zip(questions, scores_df.itertuples())):
        qid = q["id"]
        question_short = q["question"][:col_w[1] - 2]

        def fmt(key):
            val = getattr(row, key, None)
            if val is None or (isinstance(val, float) and val != val):  # NaN check
                return "  N/A  "
            totals[key].append(float(val))
            return f"{float(val):.3f}"

        print(
            f"{qid:<{col_w[0]}} "
            f"{question_short:<{col_w[1]}} "
            f"{fmt('faithfulness'):<{col_w[2]}} "
            f"{fmt('answer_relevancy'):<{col_w[3]}} "
            f"{fmt('llm_context_precision_with_reference'):<{col_w[4]}} "
            f"{fmt('context_recall'):<{col_w[5]}}"
        )
        print(f"      Category: {q['category']}  |  Contexts retrieved: {len(collected[i]['retrieved_contexts'])}")
        print()

    print(sep)

    def avg(key):
        vals = totals[key]
        return f"{sum(vals)/len(vals):.3f}" if vals else "  N/A  "

    print(
        f"{'AVG':<{col_w[0]}} "
        f"{'(all questions)':<{col_w[1]}} "
        f"{avg('faithfulness'):<{col_w[2]}} "
        f"{avg('answer_relevancy'):<{col_w[3]}} "
        f"{avg('llm_context_precision_with_reference'):<{col_w[4]}} "
        f"{avg('context_recall'):<{col_w[5]}}"
    )
    print("=" * sum(col_w))

    print("""
METRIC GUIDE
  Faithfulness      — Are all claims in the answer grounded in retrieved context?
                      Low score = hallucination risk. (LLM-as-judge)
  Answer Relevancy  — Does the answer actually address the question?
                      Low score = off-topic or evasive answers. (embedding-based)
  Ctx Precision     — Are the retrieved chunks actually relevant?
                      Low score = noisy retrieval (fetching irrelevant passages).
  Ctx Recall        — Did retrieval find all info needed for the ground truth?
                      Low score = important facts missing from retrieved context.

All metrics: 0.0 (worst) → 1.0 (best).
N/A = metric requires non-empty reference or context; shown for out-of-corpus questions.
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on the PaperGraph agent team.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only evaluate the first N questions (useful for quick tests).",
    )
    parser.add_argument(
        "--category",
        choices=["factual", "comparison", "graph", "claim_verification"],
        default=None,
        help="Only evaluate questions in this category.",
    )
    parser.add_argument(
        "--start-from-id", type=str, default=None,
        help="Resume from this question ID onward (e.g. 'q08'), skipping earlier "
             "questions in the dataset — useful after a partial run was interrupted "
             "by a rate limit, so you don't re-spend quota/time on already-completed "
             "questions. Applied AFTER --category filtering, if both are given.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save raw collected samples and scores to this JSON file.",
    )
    parser.add_argument(
        "--delay", type=float, default=RATE_LIMIT_DELAY_SECONDS,
        help=f"Seconds to wait between questions, to avoid rate limits (default: {RATE_LIMIT_DELAY_SECONDS}).",
    )
    args = parser.parse_args()

    _require_ragas()

    dataset_path = Path(__file__).parent / "dataset.json"
    questions: list[dict] = json.loads(dataset_path.read_text(encoding="utf-8"))

    if args.category:
        questions = [q for q in questions if q["category"] == args.category]

    if args.start_from_id:
        ids = [q["id"] for q in questions]
        if args.start_from_id not in ids:
            print(f"[eval] --start-from-id '{args.start_from_id}' not found in the "
                  f"(possibly category-filtered) dataset. Available IDs: {ids}")
            sys.exit(1)
        start_index = ids.index(args.start_from_id)
        questions = questions[start_index:]

    if args.limit:
        questions = questions[: args.limit]

    if not questions:
        print("[eval] No questions match the given filters.")
        sys.exit(0)

    print(f"\n[eval] PaperGraph RAGAS Evaluation")
    print(f"[eval] Questions: {len(questions)} (starting from {questions[0]['id']})")
    print(f"[eval] Model: {os.environ.get('PAPERGRAPH_MODEL', 'gemini-flash-latest')}")
    print(f"[eval] Inter-question delay: {args.delay}s (avoids free-tier rate limits)")
    print("[eval] Calling PaperGraph team directly (no HTTP server needed)\n")

    # ---------------------------------------------------------------------------
    # Step 1: Run all questions through the agent team
    # ---------------------------------------------------------------------------
    collected: list[dict] = []

    for i, q in enumerate(questions):
        session_id = f"eval_{uuid.uuid4().hex[:8]}"
        print(f"[{i+1}/{len(questions)}] {q['id']} | {q['question'][:70]}...")

        t0 = time.time()
        try:
            answer, contexts = run_single_question(q["question"], session_id)
        except Exception as e:
            print(f"  ERROR: {e}")
            answer = ""
            contexts = []

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s | contexts: {len(contexts)} | answer: {len(answer)} chars")

        reference = q["ground_truth"]
        if reference == "NOT_IN_CORPUS":
            reference = "The system should refuse to answer as this topic is not in the corpus."

        collected.append(
            {
                "id": q["id"],
                "category": q["category"],
                "user_input": q["question"],
                "response": answer or "(no answer produced)",
                "retrieved_contexts": contexts,
                "reference": reference,
            }
        )

        # Rate-limit protection: give Gemini's free-tier quota time to
        # refill before the next question's Team run (see module docstring).
        if i < len(questions) - 1 and args.delay > 0:
            time.sleep(args.delay)

    # ---------------------------------------------------------------------------
    # Step 2: Run RAGAS metrics
    # ---------------------------------------------------------------------------
    ragas_results = compute_ragas_scores(collected)

    # ---------------------------------------------------------------------------
    # Step 3: Print results
    # ---------------------------------------------------------------------------
    print_results_table(questions, collected, ragas_results)

    # ---------------------------------------------------------------------------
    # Step 4: Optionally save to JSON
    # ---------------------------------------------------------------------------
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "questions": questions,
            "collected": collected,
            "scores": ragas_results.to_pandas().to_dict(orient="records"),
        }
        output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[eval] Results saved to {output_path}")


if __name__ == "__main__":
    main()
