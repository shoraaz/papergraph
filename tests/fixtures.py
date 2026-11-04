"""
A tiny, hand-written corpus for contract tests. These are FIXTURES —
not the real PaperGraph corpus and not claims about the real papers.

Designed so the prerequisite planner has something to do:

  2310.11511v1 (target) REQUIRES rag, dense retrieval, instruction tuning
    rag             <- taught by 2005.11401v4
    dense retrieval <- taught by 2005.11401v4 and 2004.04906v3
    instruction tuning <- taught ONLY by a private upload in workspace
                          "sess_fixture_owner" (must stay invisible to
                          everyone else)
"""

from stores.base import ChunkRecord, Paper

SHARED = "shared"
OWNER = "sess_fixture_owner"
OTHER = "sess_fixture_other"

PAPERS = [
    (
        Paper("1706.03762v7", "Attention Is All You Need", SHARED, ["A. Vaswani"], "Transformers.", "2017-06-12"),
        ["sequence to sequence models", "attention mechanism"],
        [],
        [],
    ),
    (
        Paper("2004.04906v3", "Dense Passage Retrieval for Open-Domain Question Answering", SHARED, ["V. Karpukhin"], "DPR.", "2020-04-10"),
        ["dense retrieval"],
        ["attention mechanism"],
        [],
    ),
    (
        Paper("2005.11401v4", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", SHARED, ["P. Lewis"], "RAG.", "2020-05-22"),
        ["retrieval augmented generation", "dense retrieval"],
        ["sequence to sequence models"],
        ["2004.04906"],
    ),
    (
        Paper("2310.11511v1", "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection", SHARED, ["A. Asai"], "Self-RAG.", "2023-10-17"),
        ["self-reflection tokens"],
        ["retrieval augmented generation", "dense retrieval", "instruction tuning"],
        ["2005.11401", "2004.04906", "9999.99999"],
    ),
    (
        Paper("2501.00001v1", "Private Upload About Instruction Tuning", OWNER, ["Visitor"], "Upload.", "2025-01-01"),
        ["instruction tuning"],
        ["retrieval augmented generation"],
        [],
    ),
]

CHUNKS = [
    ChunkRecord("2005.11401v4", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "abstract", 0,
                "We combine a parametric seq2seq generator with a non-parametric dense vector index of Wikipedia.", SHARED),
    ChunkRecord("2005.11401v4", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "results", 1,
                "RAG achieves state of the art exact match on Natural Questions, WebQuestions and TriviaQA.", SHARED),
    ChunkRecord("2310.11511v1", "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection", "abstract", 0,
                "Self-RAG trains a model to emit self-reflection tokens that decide when to retrieve and critique.", SHARED),
    ChunkRecord("2310.11511v1", "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection", "results", 1,
                "Self-RAG outperforms ChatGPT on open-domain QA, reasoning and fact verification benchmarks.", SHARED),
    ChunkRecord("2004.04906v3", "Dense Passage Retrieval for Open-Domain Question Answering", "methods", 0,
                "A dual encoder maps questions and passages to dense vectors; retrieval is maximum inner product search.", SHARED),
    ChunkRecord("2501.00001v1", "Private Upload About Instruction Tuning", "abstract", 0,
                "zebra quokka secret upload content about instruction tuning that only its owner may see.", OWNER),
]
