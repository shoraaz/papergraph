# PaperGraph Eval — RAGAS Metrics Guide

This directory contains the evaluation harness for measuring PaperGraph's RAG quality using [RAGAS](https://docs.ragas.io).

---

## What is RAGAS?

**RAGAS** (Retrieval-Augmented Generation Assessment) is an open-source framework for objectively measuring how well a RAG pipeline is working. It separates quality into four distinct dimensions so you know *where* a system is failing.

---

## The Four Metrics

### 1. 🎯 Faithfulness (`faithfulness`)
**"Is everything the model said actually in the retrieved context?"**

- Decomposes the model's answer into atomic factual claims
- Checks each claim against the retrieved context using an LLM judge
- **High score** = the model only says things it found in context (no hallucination)
- **Low score** = the model is making things up or blending in parametric knowledge

> **Example:** If the retrieved chunk says "Self-RAG uses 4 reflection tokens" and the model says "Self-RAG uses 6 reflection tokens" — faithfulness drops.

---

### 2. 💬 Answer Relevancy (`answer_relevancy`)
**"Does the answer actually address what was asked?"**

- Reverse-generates synthetic questions from the answer using embeddings
- Measures cosine similarity between those reverse-questions and the original question
- **High score** = the answer directly addresses the question
- **Low score** = the model gave a vague, off-topic, or evasive response

> **Example:** User asks "What is Self-RAG?" and the model responds with a long tangent about CRAG — answer relevancy drops even if the answer is factually correct.

---

### 3. 🔍 Context Precision (`llm_context_precision_with_reference`)
**"Were the retrieved chunks actually useful? (no noise?)"**

- Checks what fraction of the retrieved context was genuinely needed to produce the ground-truth answer
- **High score** = retrieval is precise, pulling only relevant chunks
- **Low score** = retrieval is noisy, pulling lots of irrelevant passages alongside the good ones

> **Example:** If 8 chunks were retrieved but only 1 was about Self-RAG and the rest were about unrelated topics — precision is low even if the model still answered correctly.

---

### 4. 📚 Context Recall (`context_recall`)
**"Did retrieval find everything needed to answer correctly?"**

- Checks how much of the ground-truth answer is attributable to the retrieved context
- **High score** = the retrieved chunks contain all the information needed
- **Low score** = key facts in the ground truth are missing from the retrieved context

> **Example:** If the ground truth mentions Self-RAG's 4 specific reflection token types but none of the retrieved chunks mention those tokens — recall is low.

---

## Score Interpretation

| Score Range | Meaning |
|-------------|---------|
| 0.9 – 1.0 | Excellent |
| 0.7 – 0.9 | Good |
| 0.5 – 0.7 | Needs work |
| < 0.5 | Poor |

---

## Dataset Categories

| Category | What it tests |
|----------|--------------|
| `factual` | Direct retrieval questions about a single paper |
| `comparison` | Questions requiring synthesizing across 2+ papers |
| `graph` | Prerequisite/citation graph traversal (Neo4j) |
| `claim_verification` | Whether claims in abstracts hold up against results |

---

## Running the Evals

```powershell
# Quick test: first 3 questions only (~2 min)
.venv\Scripts\python.exe eval/run_evals.py --limit 3

# Full eval: all 13 questions (~8-12 min)
.venv\Scripts\python.exe eval/run_evals.py

# Only factual questions
.venv\Scripts\python.exe eval/run_evals.py --category factual

# Save results to JSON
.venv\Scripts\python.exe eval/run_evals.py --output eval/results.json
```

> No server needed — the eval calls `build_papergraph_team()` directly.
> Requires `GOOGLE_API_KEY` in `.env` for both PaperGraph and the RAGAS judge LLM.

---

## Agno APIs Used

The eval runner uses the non-streaming `Team.run()` API per the [Agno docs](https://docs.agno.com/teams/running-teams#run-output):

```python
# Non-streaming run — returns TeamRunOutput directly
run_output = team.run(question, stream=False)

# Final answer
answer = run_output.content

# Per-member results (Retriever, Grader, Sequencer, etc.)
for member_response in run_output.member_responses:
    # Tool call results appear as 'tool' role messages
    for msg in member_response.messages:
        if msg.role == "tool":
            # This is a retrieved chunk or graph query result
            contexts.append(msg.content)
```

`TeamRunOutput` schema: [docs.agno.com/reference/teams/team-response](https://docs.agno.com/reference/teams/team-response)

---

## Adding More Questions

Edit `eval/dataset.json` — each entry follows this schema:

```json
{
  "id": "q16",
  "question": "Your question here",
  "ground_truth": "The ideal answer — used for context recall and precision scoring",
  "relevant_paper_ids": ["2310.11511v1"],
  "category": "factual"
}
```

For out-of-corpus questions, set `"ground_truth": "NOT_IN_CORPUS"` — the runner handles this case specially.
