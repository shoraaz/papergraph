"""
stores/neo4j_store.py — the concept graph on Neo4j (system of record)
=====================================================================

Graph model (unchanged from v1):

  (:Paper {paper_id, base_id, title, authors, abstract, published_date,
           arxiv_url, workspace_id})
  (:Concept {name, display_name})
  (:Paper)-[:CITES]->(:Paper)
  (:Paper)-[:TEACHES]->(:Concept)   -- a good source to LEARN the concept
  (:Paper)-[:REQUIRES]->(:Concept)  -- assumes the reader already knows it

What changed from v1, and why:

  * Agents no longer write Cypher. v1 handed the LLM a raw Cypher tool
    and asked it, in the prompt, to add `WHERE p.workspace_id IN [...]`
    to every query. A prompt is not an access-control boundary: a crafted
    question could read other sessions' uploads, run an unbounded
    traversal, or — since the tool could write — delete the graph. Every
    query now lives here, is parameterized (no string interpolation of
    user input), takes `workspaces` from the server, and runs in a READ
    transaction. Only the ETL calls `upsert_paper`.
  * Async driver, one pool per process, created once at startup.
  * Citation targets are resolved by `base_id` so a CITES edge to
    "2005.11401" connects to the ingested "2005.11401v4" (in v1 they
    never connected).
  * Re-ingesting a paper replaces its TEACHES/REQUIRES edges instead of
    accumulating stale ones.
"""

from __future__ import annotations

from typing import Optional, Sequence

from neo4j import READ_ACCESS, AsyncDriver, AsyncGraphDatabase

from config import SHARED_WORKSPACE
from stores.base import (
    Concept,
    ConceptEdge,
    GraphSnapshot,
    Paper,
    PrereqInputs,
    base_arxiv_id,
    json_safe,
    looks_like_arxiv_id,
    normalize_concept,
    normalize_paper_ref,
    title_words,
)

SCHEMA_CYPHER = [
    "CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
    "CREATE CONSTRAINT concept_name_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
    "CREATE INDEX paper_title_idx IF NOT EXISTS FOR (p:Paper) ON (p.title)",
    "CREATE INDEX paper_workspace_idx IF NOT EXISTS FOR (p:Paper) ON (p.workspace_id)",
    "CREATE INDEX paper_base_idx IF NOT EXISTS FOR (p:Paper) ON (p.base_id)",
]

_PAPER_FIELDS = ("paper_id", "title", "workspace_id", "authors", "abstract", "published_date", "arxiv_url")


def _paper(props: dict) -> Paper:
    props = json_safe(dict(props or {}))
    return Paper(**{k: props.get(k) or ([] if k == "authors" else "") for k in _PAPER_FIELDS})


class Neo4jGraphStore:
    read_only = False

    def __init__(self, uri: str, user: str, password: str, database: Optional[str] = None):
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_pool_size=50,
            connection_acquisition_timeout=10,
        )
        self._db = database

    # -- plumbing ------------------------------------------------------------

    async def _read(self, query: str, **params) -> list[dict]:
        async def work(tx):
            result = await tx.run(query, **params)
            return await result.data()

        async with self._driver.session(database=self._db, default_access_mode=READ_ACCESS) as session:
            return await session.execute_read(work)

    async def ensure_ready(self) -> None:
        await self._driver.verify_connectivity()
        async with self._driver.session(database=self._db) as session:
            for stmt in SCHEMA_CYPHER:
                await session.run(stmt)

    async def close(self) -> None:
        await self._driver.close()

    async def _resolve(self, paper_ref: str, workspaces: Sequence[str]) -> Optional[Paper]:
        ref = normalize_paper_ref(paper_ref)
        if not ref:
            return None
        rows = await self._read(
            """
            MATCH (p:Paper)
            WHERE p.workspace_id IN $ws AND p.paper_id STARTS WITH $ref
              AND p.title IS NOT NULL AND p.title <> ''
            RETURN p {.*} AS p ORDER BY p.paper_id DESC LIMIT 1
            """,
            ws=list(workspaces),
            ref=ref,
        )
        return _paper(rows[0]["p"]) if rows else None

    # -- reads ---------------------------------------------------------------

    async def find_papers(self, text: str, workspaces: Sequence[str], limit: int = 5) -> list[Paper]:
        if looks_like_arxiv_id(text):
            rows = await self._read(
                """
                MATCH (p:Paper)
                WHERE p.workspace_id IN $ws AND p.paper_id STARTS WITH $ref
                  AND p.title IS NOT NULL AND p.title <> ''
                RETURN p {.*} AS p ORDER BY p.title LIMIT $limit
                """,
                ws=list(workspaces),
                ref=normalize_paper_ref(text),
                limit=limit,
            )
        else:
            q = (text or "").strip().lower()
            if not q:
                return []
            rows = await self._read(
                """
                MATCH (p:Paper)
                WHERE p.workspace_id IN $ws AND p.title IS NOT NULL AND p.title <> ''
                  AND (toLower(p.title) CONTAINS $q
                       OR (size($words) > 0 AND ALL(w IN $words WHERE toLower(p.title) CONTAINS w)))
                RETURN p {.*} AS p ORDER BY p.title LIMIT $limit
                """,
                ws=list(workspaces),
                q=q,
                words=title_words(q),
                limit=limit,
            )
        return [_paper(r["p"]) for r in rows]

    async def get_paper(self, paper_ref: str, workspaces: Sequence[str]) -> Optional[dict]:
        p = await self._resolve(paper_ref, workspaces)
        if p is None:
            return None
        rows = await self._read(
            """
            MATCH (p:Paper {paper_id: $pid})
            OPTIONAL MATCH (p)-[:TEACHES]->(t:Concept)
            WITH p, collect(DISTINCT t.display_name) AS teaches
            OPTIONAL MATCH (p)-[:REQUIRES]->(r:Concept)
            WITH p, teaches, collect(DISTINCT r.display_name) AS requires
            OPTIONAL MATCH (p)-[:CITES]->(c:Paper)
              WHERE c.workspace_id IN $ws AND c.title IS NOT NULL AND c.paper_id <> p.paper_id
            WITH p, teaches, requires, collect(DISTINCT c {.paper_id, .title}) AS cites
            OPTIONAL MATCH (b:Paper)-[:CITES]->(p)
              WHERE b.workspace_id IN $ws AND b.title IS NOT NULL AND b.paper_id <> p.paper_id
            RETURN teaches, requires, cites, collect(DISTINCT b {.paper_id, .title}) AS cited_by
            """,
            pid=p.paper_id,
            ws=list(workspaces),
        )
        row = rows[0] if rows else {"teaches": [], "requires": [], "cites": [], "cited_by": []}
        return {
            **p.summary(),
            "authors": p.authors,
            "abstract": p.abstract,
            "teaches": sorted(x for x in row["teaches"] if x),
            "requires": sorted(x for x in row["requires"] if x),
            "cites_in_corpus": sorted(row["cites"], key=lambda d: d["paper_id"]),
            "cited_by_in_corpus": sorted(row["cited_by"], key=lambda d: d["paper_id"]),
        }

    async def prerequisite_inputs(self, paper_ref: str, workspaces: Sequence[str]) -> PrereqInputs:
        target = await self._resolve(paper_ref, workspaces)
        if target is None:
            return PrereqInputs(paper=None, required=[], teachers={}, teacher_edges=[])
        rows = await self._read(
            """
            MATCH (p:Paper {paper_id: $pid})-[:REQUIRES]->(c:Concept)
            OPTIONAL MATCH (t:Paper)-[:TEACHES]->(c)
              WHERE t.workspace_id IN $ws AND t.paper_id <> $pid
                AND t.title IS NOT NULL AND t.title <> ''
            RETURN c.name AS name, c.display_name AS display_name, collect(DISTINCT t {.*}) AS teachers
            ORDER BY name
            """,
            pid=target.paper_id,
            ws=list(workspaces),
        )
        required = [Concept(name=r["name"], display_name=r["display_name"] or r["name"]) for r in rows]
        teachers = {
            r["name"]: sorted((_paper(t) for t in r["teachers"]), key=lambda p: p.paper_id) for r in rows
        }
        teacher_ids = sorted({p.paper_id for ps in teachers.values() for p in ps})
        edge_rows = await self._read(
            """
            MATCH (t:Paper)-[r:TEACHES|REQUIRES]->(c:Concept)
            WHERE t.paper_id IN $ids
            RETURN t.paper_id AS paper_id, type(r) AS rel, c.name AS concept
            ORDER BY paper_id, rel DESC, concept
            """,
            ids=teacher_ids,
        )
        edges = [ConceptEdge(r["paper_id"], r["rel"], r["concept"]) for r in edge_rows]
        return PrereqInputs(paper=target, required=required, teachers=teachers, teacher_edges=edges)

    async def papers_teaching(self, concept: str, workspaces: Sequence[str], limit: int = 8) -> list[dict]:
        needle = normalize_concept(concept)
        if not needle:
            return []
        return await self._read(
            """
            MATCH (p:Paper)-[:TEACHES]->(c:Concept)
            WHERE p.workspace_id IN $ws AND p.title IS NOT NULL AND p.title <> ''
              AND (c.name CONTAINS $needle OR $needle CONTAINS c.name)
            RETURN c.display_name AS concept, p.paper_id AS paper_id, p.title AS title
            ORDER BY c.name, p.paper_id LIMIT $limit
            """,
            ws=list(workspaces),
            needle=needle,
            limit=limit,
        )

    async def list_papers(self, workspaces: Sequence[str]) -> list[Paper]:
        rows = await self._read(
            """
            MATCH (p:Paper)
            WHERE p.workspace_id IN $ws AND p.title IS NOT NULL AND p.title <> ''
            RETURN p {.*} AS p ORDER BY p.workspace_id <> $shared, p.title
            """,
            ws=list(workspaces),
            shared=SHARED_WORKSPACE,
        )
        return [_paper(r["p"]) for r in rows]

    async def concept_edges_for(self, paper_ids: Sequence[str]) -> list[tuple[ConceptEdge, Concept]]:
        rows = await self._read(
            """
            MATCH (p:Paper)-[r:TEACHES|REQUIRES]->(c:Concept)
            WHERE p.paper_id IN $ids
            RETURN p.paper_id AS paper_id, type(r) AS rel, c.name AS name, c.display_name AS display_name
            ORDER BY paper_id, rel DESC, name
            """,
            ids=list(paper_ids),
        )
        return [
            (ConceptEdge(r["paper_id"], r["rel"], r["name"]), Concept(r["name"], r["display_name"] or r["name"]))
            for r in rows
        ]

    # -- write path (ETL only) -----------------------------------------------

    async def upsert_paper(
        self, paper: Paper, teaches: Sequence[str], requires: Sequence[str], cites: Sequence[str]
    ) -> None:
        base_id = base_arxiv_id(paper.paper_id)

        def concepts(names: Sequence[str]) -> list[dict]:
            seen, out = set(), []
            for n in names:
                key = normalize_concept(n)
                if key and key not in seen:
                    seen.add(key)
                    out.append({"name": key, "display": n.strip()})
            return out

        cite_bases = sorted({base_arxiv_id(c) for c in cites} - {base_id})

        async def work(tx):
            await tx.run(
                """
                MERGE (p:Paper {paper_id: $paper_id})
                SET p.base_id = $base_id, p.title = $title, p.authors = $authors,
                    p.abstract = $abstract, p.published_date = $published_date,
                    p.arxiv_url = $arxiv_url, p.workspace_id = $workspace_id
                """,
                paper_id=paper.paper_id,
                base_id=base_id,
                title=paper.title,
                authors=paper.authors,
                abstract=paper.abstract,
                published_date=paper.published_date,
                arxiv_url=paper.arxiv_url,
                workspace_id=paper.workspace_id,
            )
            # A version-less stub created earlier by someone citing us:
            # move its incoming CITES onto the real node, then drop it.
            await tx.run(
                """
                MATCH (stub:Paper {paper_id: $base_id})
                WHERE stub.title IS NULL AND $base_id <> $paper_id
                MATCH (p:Paper {paper_id: $paper_id})
                OPTIONAL MATCH (src:Paper)-[:CITES]->(stub)
                FOREACH (_ IN CASE WHEN src IS NULL THEN [] ELSE [1] END | MERGE (src)-[:CITES]->(p))
                WITH DISTINCT stub
                DETACH DELETE stub
                """,
                base_id=base_id,
                paper_id=paper.paper_id,
            )
            await tx.run(
                "MATCH (p:Paper {paper_id: $pid})-[r:TEACHES|REQUIRES]->(:Concept) DELETE r",
                pid=paper.paper_id,
            )
            for rel, items in (("TEACHES", concepts(teaches)), ("REQUIRES", concepts(requires))):
                await tx.run(
                    f"""
                    UNWIND $items AS item
                    MERGE (c:Concept {{name: item.name}})
                      ON CREATE SET c.display_name = item.display
                    WITH c
                    MATCH (p:Paper {{paper_id: $pid}})
                    MERGE (p)-[:{rel}]->(c)
                    """,  # rel is one of two constants above, never user input
                    items=items,
                    pid=paper.paper_id,
                )
            if cite_bases:
                existing = await tx.run(
                    "MATCH (e:Paper) WHERE e.base_id IN $bases AND e.title IS NOT NULL "
                    "RETURN e.base_id AS base, e.paper_id AS pid",
                    bases=cite_bases,
                )
                resolved = {r["base"]: r["pid"] for r in await existing.data()}
                targets = [resolved.get(b, b) for b in cite_bases]
                await tx.run(
                    """
                    MATCH (p:Paper {paper_id: $pid})
                    UNWIND $targets AS tid
                    MERGE (to:Paper {paper_id: tid})
                    MERGE (p)-[:CITES]->(to)
                    """,
                    pid=paper.paper_id,
                    targets=targets,
                )

        async with self._driver.session(database=self._db) as session:
            await session.execute_write(work)

    async def export_snapshot(self, workspaces: Sequence[str] = (SHARED_WORKSPACE,)) -> GraphSnapshot:
        """Only the shared corpus by default — visitors' uploads must never
        end up in a snapshot that gets committed and served publicly."""
        ws = list(workspaces)
        papers = [
            _paper(r["p"])
            for r in await self._read(
                "MATCH (p:Paper) WHERE p.workspace_id IN $ws AND p.title IS NOT NULL AND p.title <> '' "
                "RETURN p {.*} AS p ORDER BY p.paper_id",
                ws=ws,
            )
        ]
        ids = [p.paper_id for p in papers]
        edge_rows = await self.concept_edges_for(ids)
        concepts = {c.name: c for _, c in edge_rows}
        cite_rows = await self._read(
            "MATCH (a:Paper)-[:CITES]->(b:Paper) WHERE a.paper_id IN $ids "
            "RETURN a.paper_id AS a, b.paper_id AS b ORDER BY a, b",
            ids=ids,
        )
        return GraphSnapshot(
            papers=papers,
            concepts=sorted(concepts.values(), key=lambda c: c.name),
            concept_edges=[e for e, _ in edge_rows],
            cites=[(r["a"], r["b"]) for r in cite_rows],
        )
