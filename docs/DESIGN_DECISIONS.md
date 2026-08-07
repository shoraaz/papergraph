# PaperGraph Architectural Design & Rationale

This document consolidates key architectural decisions, empirical findings, and design trade-offs across PaperGraph.

---

## 1. Agent Design & Self-RAG Enforcement (`agents.py`)

### 1.1 Team Architecture & Role Division
PaperGraph uses an Agno `Team` comprising specialized sub-agents led by a Team Leader:
* **RetrieverAgent**: Agentic RAG dispatcher deciding between Qdrant vector search and Neo4j Cypher queries.
* **GraderAgent**: Self-RAG checker returning a typed `GroundingCheck` schema (relevance, confidence, missing information).
* **SequencerAgent**: GraphRAG specialist running Cypher traversals and topological sorting over `REQUIRES` concept edges.
* **JargonAgent**: Paragraph-by-paragraph plain-English explanations.
* **ClaimVerifierAgent**: Corrective RAG verifier scoring paper abstract/intro claims against experimental results (`ClaimCheck`).
* **PaperGraphTeam**: Leader orchestrating delegation, session memory (Mem0), and final synthesis.

### 1.2 Mandatory Grader Delegation
* **Pattern**: Self-RAG requires an explicit grading step.
* **Finding**: The Team leader sometimes skipped delegating to Grader if Retriever's response appeared confident.
* **Decision**: Team instructions strictly enforce mandatory delegation (`"for EVERY content query"` and `"never skip"`).
* **Context Sharing**: `share_member_interactions=True` is enabled so member responses (like Retriever's output) are passed verbatim to Grader rather than being summarized by the leader.

### 1.3 Honest Refusals & Guardrail Rules
* **Strict Rule**: When context is absent or incomplete, agents MUST NOT fall back to general/parametric LLM knowledge.
* **Refusal Rule**: An empty query result must produce an explicit refusal ("This is not in my corpus") and nothing else. Adding general knowledge after a refusal is strictly prohibited.

### 1.4 Schema Grounding & Nuanced Graph Responses
* **Schema Grounding**: Neo4j properties (`paper_id`, `title`, `workspace_id`) are explicitly documented in agent instructions. arXiv IDs use version suffixes (e.g. `2502.01113v3`), so agents are instructed to use `STARTS WITH` queries rather than exact equality.
* **Partial Findings**: If a paper and its concepts exist in the graph, but no in-corpus paper teaches those concepts, the Sequencer agent reports partial findings rather than collapsing the response into a flat refusal.

### 1.5 Three-Way Claim Check Verdicts
* `ClaimCheck` supports three states instead of a boolean pair:
  1. `is_supported`: Abstract claim aligns with experimental data.
  2. `contradiction_found`: Results directly contradict the claim.
  3. `overstated`: Claim is accurate but broader than experimental evidence supports (e.g., tested on one dataset/model).

---

## 2. Multi-Store Data Infrastructure (`db.py`)

### 2.1 Tri-Store Specialization
PaperGraph leverages three open-source data stores:
1. **Neo4j**: Graph structure storing paper-to-paper citation (`CITES`) and paper-to-concept (`TEACHES`/`REQUIRES`) edges.
2. **Qdrant**: Dense/sparse hybrid vector store with BAAI/bge-base-en-v1.5 embeddings and cross-encoder reranking (`BAAI/bge-reranker-v2-m3`).
3. **Mem0**: Local session memory tracking conversation context per session UUID.

### 2.2 Workspace Isolation
Multi-tenant uploads isolate visitor data:
* **Base Corpus**: Tagged with `workspace_id = "shared"`.
* **User Uploads**: Tagged with `workspace_id = <session_id>`.
* **Qdrant Isolation**: Hard server-side enforcement via custom `WorkspaceScopedQdrant` wrapper overriding filter generation.
* **Neo4j Isolation**: Cypher queries filter on `p.workspace_id IN [$workspace_id, 'shared']`.

---

## 3. Real-Time Streaming & Event Pipeline (`server.py` & `event_mapper.py`)

### 3.1 Thread-Safe Event Queue for SSE
* **Challenge**: Non-blocking SSE streaming requires incremental delivery, but Agno's `team.run(stream=True)` runs synchronously.
* **Solution**: A background worker thread executes `team.run(...)` and pushes raw events to a `queue.Queue`. The async FastAPI event loop consumes items using `asyncio.to_thread(event_queue.get)` and yields formatted Server-Sent Events (SSE) in real time.

### 3.2 BYOK (Bring Your Own Key) Model Handling
* Requests can supply custom LLM credentials (`provider`, `model_id`, `api_key`).
* `ModelChoice` objects thread credentials per-request without persisting keys to disk or environment variables.

---

## 4. Ingestion & ETL Pipeline (`etl.py`)

### 4.1 Section-Aware Chunking
* `SectionAwareChunking` tags chunks with section headers (Abstract, Intro, Results, Methods).
* Enables `ClaimVerifierAgent` to isolate results sections when auditing abstract claims.

### 4.2 Concept Extraction & Graph Upsert
* Uses LLM calls to extract key concepts taught or required by a paper.
* Upserts nodes and edges into Neo4j using idempotent Cypher queries (`MERGE`).
* Normalizes arXiv queries by enforcing exact-phrase matching (`abs:"..."`) to avoid loosely related matches.
