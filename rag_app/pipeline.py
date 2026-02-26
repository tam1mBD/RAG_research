"""
pipeline.py — Full end-to-end RAG pipeline.

Single entry point: RAGPipeline.ask(query) → RAGResponse

Internal flow
─────────────
user query (str)
  │
  ▼
[1] embed_text()          — cached embedding of the query
  │
  ▼
[2] VectorStore.query()   — cosine similarity search, top-k chunks
  │
  ▼
[3] PromptBuilder.build() — 5-layer prompt assembled within token budget
  │
  ▼
[4] LLMClient.generate()  — LangChain ChatOpenAI, temp=0, grounding check
  │
  ▼
[5] CitationMapper.map()  — resolve [Doc N] refs → source file + snippet
  │
  ▼
RAGResponse                — answer, bibliography, warnings, metadata

Public API
──────────
RAGPipeline
    .ask(query, top_k, threshold) -> RAGResponse
    .index_status()               -> dict
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

from rag_app.embedder import embed_text
from rag_app.vector_store import VectorStore, SearchResult
from rag_app.prompt_builder import PromptBuilder
from rag_app.llm_client import LLMClient, LLMResponse
from rag_app.citation_mapper import CitationMapper, Citation, MappingResult
from rag_app.config import TOP_K_RESULTS, SIMILARITY_THRESHOLD
from rag_app.logger import get_logger

log = get_logger(__name__)


# ── Response dataclass ────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    """Complete output of one RAG pipeline run."""

    query: str                                   # original user question

    # Core answer
    answer: str                                  # raw LLM answer
    annotated_answer: str                        # answer with enriched inline citations
    bibliography: str                            # formatted reference block

    # Retrieved context
    results: List[SearchResult] = field(default_factory=list)
    citations: List[Citation]   = field(default_factory=list)
    unresolved_citations: List[int] = field(default_factory=list)

    # Quality flags
    is_refused: bool  = False
    is_grounded: bool = True
    warnings: List[str] = field(default_factory=list)

    # Token & latency metadata
    prompt_tokens: int     = 0
    completion_tokens: int = 0
    total_tokens: int      = 0
    latency_seconds: float = 0.0

    def __str__(self) -> str:           # quick human-readable summary
        lines = [
            f"Query    : {self.query}",
            f"Refused  : {self.is_refused}",
            f"Grounded : {self.is_grounded}",
            f"Tokens   : {self.total_tokens} ({self.prompt_tokens}p + {self.completion_tokens}c)",
            f"Latency  : {self.latency_seconds:.2f}s",
            "",
            "── Answer ──",
            self.annotated_answer,
        ]
        if self.bibliography:
            lines += ["", self.bibliography]
        if self.warnings:
            lines += ["", "── Warnings ──"] + [f"  ⚠ {w}" for w in self.warnings]
        return "\n".join(lines)


# ── RAGPipeline ───────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Orchestrates the full retrieval-augmented generation pipeline.

    Instantiate once and call .ask() repeatedly — all components are
    reused across calls, saving repeated initialisation overhead.

    Parameters
    ----------
    store   : optional pre-built VectorStore (created fresh if None).
    """

    def __init__(self, store: VectorStore | None = None) -> None:
        self._store   = store or VectorStore()
        self._builder = PromptBuilder()
        self._llm     = LLMClient()
        self._mapper  = CitationMapper()
        log.info("RAGPipeline ready.")

    # ── Public ────────────────────────────────────────────────────────────────

    def ask(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> RAGResponse:
        """
        Run the full RAG pipeline for *query*.

        Parameters
        ----------
        query     : user's natural-language question.
        top_k     : max number of chunks to retrieve.
        threshold : minimum similarity score to include a chunk.

        Returns
        -------
        RAGResponse — fully populated result object.
        """
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty.")

        t_start = time.perf_counter()
        log.info("=" * 60)
        log.info("RAG query: %s", query)

        # ── Step 1: Embed query ───────────────────────────────────────────────
        log.info("[1/5] Embedding query…")
        query_embedding = embed_text(query)

        # ── Step 2: Retrieve top-k chunks ────────────────────────────────────
        log.info("[2/5] Retrieving top-%d chunks (threshold=%.2f)…", top_k, threshold)
        results: List[SearchResult] = self._store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            threshold=threshold,
        )

        if not results:
            log.warning("No relevant chunks retrieved — pipeline will issue refusal.")

        # ── Step 3: Build multi-layer prompt ─────────────────────────────────
        log.info("[3/5] Building prompt (%d chunks)…", len(results))
        messages = self._builder.build(query=query, results=results)

        # ── Step 4: LLM generation ────────────────────────────────────────────
        log.info("[4/5] Generating answer…")
        context_sources = [r.source for r in results]
        llm_resp: LLMResponse = self._llm.generate(
            messages=messages,
            context_sources=context_sources,
        )

        # ── Step 5: Citation mapping ─────────────────────────────────────────
        log.info("[5/5] Mapping citations…")
        mapping: MappingResult = self._mapper.map(
            answer=llm_resp.answer,
            results=results,
        )
        bibliography = CitationMapper.format_bibliography(mapping.citations)

        latency = round(time.perf_counter() - t_start, 3)
        log.info("Pipeline complete in %.2fs.", latency)

        return RAGResponse(
            query=query,
            answer=llm_resp.answer,
            annotated_answer=mapping.annotated_answer,
            bibliography=bibliography,
            results=results,
            citations=mapping.citations,
            unresolved_citations=mapping.unresolved,
            is_refused=llm_resp.is_refused,
            is_grounded=llm_resp.is_grounded,
            warnings=llm_resp.warnings,
            prompt_tokens=llm_resp.prompt_tokens,
            completion_tokens=llm_resp.completion_tokens,
            total_tokens=llm_resp.total_tokens,
            latency_seconds=latency,
        )

    def index_status(self) -> dict:
        """
        Return a summary dict of the current vector store state.

        Useful for displaying in the UI sidebar.
        """
        docs = self._store.list_documents()
        return {
            "total_chunks": self._store.count(),
            "total_documents": len(docs),
            "documents": docs,
        }
