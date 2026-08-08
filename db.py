"""
db.py — PaperGraph Data Layer
=============================

Specialized tri-store data infrastructure:
  1. Neo4j  : Graph store for citation (CITES) & concept (TEACHES/REQUIRES) edges.
  2. Qdrant : Dense/sparse hybrid vector store with reranking for chunk embeddings.
  3. Mem0   : Local session and durable user conversation memory.

Includes local embedding (BAAI/bge-base-en-v1.5) and cross-encoder reranking.
For workspace isolation details and architecture rationale, see `docs/DESIGN_DECISIONS.md`.
"""

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()  # reads .env if present; real env vars still take precedence

from agno.knowledge.knowledge import Knowledge
from agno.vectordb.qdrant import Qdrant
from agno.vectordb.search import SearchType
from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from qdrant_client.models import PayloadSchemaType

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
# AuraDB's downloaded credentials file uses NEO4J_USERNAME (not NEO4J_USER) —
# support both so either naming convention in .env works.
NEO4J_USER = os.environ.get("NEO4J_USERNAME") or os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE")  # AuraDB sets this; None = server default

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")  # None for local Docker
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "papergraph_chunks")

EMBEDDER_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDER_DIMENSIONS = 768

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

SHARED_WORKSPACE = "shared"  # the base corpus every session can see

# Payload fields we filter on and therefore need explicit Qdrant indexes
# for — Qdrant returns a 400 on any filtered query against an un-indexed
# field, rather than just being slow, so this isn't optional. content_hash
# is used internally by Agno's skip_if_exists check; paper_id and section
# are used by our own agents/queries (e.g. ClaimVerifierAgent restricting
# search to a paper's results section). These three were indexed on their
# bare (unprefixed) names and that has worked correctly in testing.
FILTERABLE_PAYLOAD_FIELDS = ["content_hash", "paper_id", "section"]

# workspace_id needs its index under the "meta_data." prefix explicitly.
# WorkspaceScopedQdrant's _format_filters() (below) builds its
# FieldCondition key as "meta_data.workspace_id" directly, matching how
# Agno's own _format_filters auto-prefixes bare field names — so the
# index must be created at that exact nested path or Qdrant returns a
# 400 "index required but not found." Confirmed via a live test: an
# index on bare "workspace_id" did NOT satisfy this filter; the nested
# path did.
WORKSPACE_ID_INDEX_FIELD = "meta_data.workspace_id"


# ---------------------------------------------------------------------------
# Neo4j — the graph itself
# ---------------------------------------------------------------------------

# Constraints/indexes. Cypher, run once at startup — idempotent (IF NOT EXISTS).
SCHEMA_CYPHER = [
    "CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
    "CREATE CONSTRAINT concept_name_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
    "CREATE INDEX paper_title_idx IF NOT EXISTS FOR (p:Paper) ON (p.title)",
    "CREATE INDEX paper_workspace_idx IF NOT EXISTS FOR (p:Paper) ON (p.workspace_id)",
]

# Node/relationship shape (for reference — not executed, just documents the model):
#
#   (:Paper {paper_id, title, authors, abstract, published_date, arxiv_url, workspace_id})
#   (:Concept {name, display_name})
#
#   (:Paper)-[:CITES]->(:Paper)
#   (:Paper)-[:TEACHES]->(:Concept)     -- this paper is a good source to LEARN this concept from
#   (:Paper)-[:REQUIRES]->(:Concept)    -- this paper ASSUMES the reader already knows this concept
#
# Reading-order sequencing (graph.py) is:
#   MATCH (p:Paper {paper_id: $target})-[:REQUIRES]->(c:Concept)
#   MATCH (source:Paper)-[:TEACHES]->(c)
#   ... then topologically order the resulting concept set by their own
#   REQUIRES dependencies, and map each concept back to its best TEACHES paper.
#
# Concept nodes are intentionally NOT workspace-scoped — "graph neural
# networks" means the same thing regardless of who uploaded the paper
# that introduced it. Only Paper nodes carry workspace_id; this lets a
# user's own uploaded paper connect into the shared concept graph
# (e.g. sharing a REQUIRES edge to a concept the base corpus also
# teaches) rather than living in total isolation.


_neo4j_driver = None


def get_neo4j_driver():
    """Returns a singleton Neo4j driver. Reuses existing connection pool across requests."""
    global _neo4j_driver
    if _neo4j_driver is None:
        _neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _neo4j_driver


def close_neo4j_driver():
    """Closes the singleton Neo4j driver connection pool if open."""
    global _neo4j_driver
    if _neo4j_driver is not None:
        _neo4j_driver.close()
        _neo4j_driver = None


def init_graph_schema() -> None:
    """Create constraints/indexes if they don't exist. Safe to re-run."""
    driver = get_neo4j_driver()
    driver.verify_connectivity()
    with driver.session(database=NEO4J_DATABASE) as session:
        for statement in SCHEMA_CYPHER:
            session.run(statement)
    print(f"[db] Neo4j schema ready at {NEO4J_URI} (user={NEO4J_USER})")



# ---------------------------------------------------------------------------
# Qdrant — vector store for chunks, with enforced workspace isolation
# ---------------------------------------------------------------------------

def get_embedder() -> Any:
    """Local BGE embedder — lazy loaded so server startup on Render is instant (<0.1s)."""
    from agno.knowledge.embedder.sentence_transformer import SentenceTransformerEmbedder
    return SentenceTransformerEmbedder(
        id=EMBEDDER_MODEL,
        dimensions=EMBEDDER_DIMENSIONS,
    )


def get_reranker() -> Any:
    """
    Local cross-encoder reranker — lazy loaded.
    Set ENABLE_RERANKER=false in env (e.g. on 512MB free-tier hosting like Render)
    to skip loading the heavy ~600MB cross-encoder model and use native Qdrant hybrid search.
    """
    if os.environ.get("ENABLE_RERANKER", "true").lower() not in ("true", "1", "yes"):
        print("[db] ENABLE_RERANKER is disabled — using native Qdrant hybrid search.")
        return None

    try:
        from agno.knowledge.reranker.sentence_transformer import SentenceTransformerReranker
        return SentenceTransformerReranker(model=RERANKER_MODEL)
    except Exception as e:
        print(f"[db] Warning: Could not load local reranker ({e}). Falling back to native Qdrant hybrid search.")
        return None


class WorkspaceScopedQdrant(Qdrant):
    """
    A Qdrant vector_db that transparently enforces a workspace filter on
    every search() / async_search() call, regardless of what filter (if
    any) the caller passes.

    Why this exists instead of relying on agents.py instructing the LLM
    to pass a filter itself: KnowledgeTools.search_knowledge (the tool
    our Retriever/ClaimVerifier agents actually call) takes only a
    `query` string — there's no filter parameter for the LLM to set or
    forget to set. Enforcing isolation here, below the tool-call layer,
    means it holds regardless of what the LLM does or doesn't do — a
    real security/correctness boundary, not just a prompt convention.

    IMPORTANT — overrides _format_filters, not search()/async_search():
    Agno's Qdrant._format_filters() only ever builds a Qdrant MatchValue
    (single exact-value equality) per key — it has no support for a
    list-of-allowed-values ("IN") filter, which is exactly what workspace
    isolation needs ("shared" OR the caller's own workspace_id). Passing
    a list straight through crashes with a pydantic ValidationError (a
    MatchValue rejects list input outright — confirmed via a live test
    while building this). So this subclass overrides _format_filters
    itself: any caller-supplied dict filters still go through the
    parent's normal exact-match logic, and the workspace clause is added
    separately as a proper Qdrant MatchAny condition, then both are
    combined into one models.Filter with an AND (must=[...]).

    allowed_workspaces defaults to [SHARED_WORKSPACE] only — pass the
    caller's own workspace_id too so their own uploads are visible
    alongside the shared base corpus.
    """

    def __init__(self, *args, allowed_workspaces: Optional[List[str]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.allowed_workspaces = list(allowed_workspaces or [SHARED_WORKSPACE])

    def _format_filters(self, filters: Optional[Dict[str, Any]]):
        conditions = []

        base_filter = super()._format_filters(filters or {})
        if base_filter is not None:
            conditions.append(base_filter)

        conditions.append(
            models.FieldCondition(
                key=WORKSPACE_ID_INDEX_FIELD,
                match=models.MatchAny(any=self.allowed_workspaces),
            )
        )

        return models.Filter(must=conditions)


def get_vector_db(workspace_id: Optional[str] = None) -> Qdrant:
    """
    Qdrant vector DB handle for chunk embeddings, scoped to a workspace.

    workspace_id=None (default): shared corpus only — used by etl.py's
    CLI ingestion of the base 18-paper corpus, and by any caller that
    genuinely wants the shared-only view.
    workspace_id=<session_id>: that session's own uploads ARE visible
    alongside the shared corpus (both allowed_workspaces entries).

    IMPORTANT: embedder must be passed into the constructor, not attached
    afterward (vector_db.embedder = ...). Agno's Qdrant reads
    self.embedder.dimensions at __init__ time to decide the collection's
    vector size when it first creates the collection — attaching the
    embedder after construction leaves self.dimensions as None, and a
    fresh collection silently gets created at Qdrant's fallback of 1536
    dims instead of our actual 768. (Hit this exact bug during initial
    ETL testing — a mismatched collection has to be deleted and
    recreated to fix, so get this right rather than attach-after.)

    reranker=get_reranker() adds the cross-encoder reranking pass on top
    of hybrid search's fused candidates — see module docstring's
    RERANKING section for why this was added.

    Local dev:            QDRANT_URL=http://localhost:6333, no API key.
    Qdrant Cloud (free):   QDRANT_URL=<cluster url>, QDRANT_API_KEY set.
    Self-hosted (final):   same class, point QDRANT_URL at your own instance.
    """
    allowed = [SHARED_WORKSPACE] if workspace_id is None else [SHARED_WORKSPACE, workspace_id]
    return WorkspaceScopedQdrant(
        collection=QDRANT_COLLECTION,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        search_type=SearchType.hybrid,  # dense + sparse — helps on jargon/acronyms
        embedder=get_embedder(),
        reranker=get_reranker(),
        allowed_workspaces=allowed,
    )


def ensure_payload_indexes() -> None:
    """
    Qdrant requires an explicit payload index on any field used in a
    filter — without one, a filtered query throws a 400 rather than
    just running unindexed. This bit us three times now during testing
    (content_hash, paper_id, and workspace_id all separately) so every
    known-filterable field is indexed here, once, up front. Safe to
    call repeatedly — re-creating an existing index just raises, which
    we log and ignore.
    """
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    try:
        for field in FILTERABLE_PAYLOAD_FIELDS + [WORKSPACE_ID_INDEX_FIELD]:
            try:
                client.create_payload_index(
                    collection_name=QDRANT_COLLECTION,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                print(f"[db] created payload index on '{field}'")
            except Exception as e:
                print(f"[db] payload index '{field}': {e}")
    finally:
        client.close()


def get_knowledge(workspace_id: Optional[str] = None) -> Knowledge:
    """The Agno Knowledge object agents attach to for agentic RAG over chunks.

    workspace_id: see get_vector_db(). None = shared-corpus-only view
    (used by etl.py's base-corpus ingestion and by CLI testing); pass
    the session's id to also surface that session's own uploads.
    """
    return Knowledge(
        name="PaperGraph Corpus",
        description="Ingested arXiv papers, chunked and embedded for semantic retrieval.",
        vector_db=get_vector_db(workspace_id=workspace_id),
    )


# ---------------------------------------------------------------------------
# CLI entrypoint: `python db.py` verifies both stores are reachable
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_graph_schema()
    kb = get_knowledge()
    ensure_payload_indexes()
    print(f"[db] Qdrant collection '{QDRANT_COLLECTION}' ready at {QDRANT_URL}")
    print(f"[db] embedder: {EMBEDDER_MODEL} ({EMBEDDER_DIMENSIONS}-dim)")
    print(f"[db] reranker: {RERANKER_MODEL} (cross-encoder, local)")
    print("[db] Mem0 memory is configured per-agent in agents.py (local instance, no server needed)")
