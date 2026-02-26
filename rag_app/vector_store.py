"""
vector_store.py — ChromaDB wrapper for storing and retrieving document chunks.

Responsibilities
----------------
- Persist embeddings + metadata to disk via ChromaDB's persistent client.
- Add chunks (with pre-computed embeddings) to the collection.
- Similarity search: returns top-k chunks above a score threshold.
- Delete all chunks that belong to a specific source document.
- List every unique source document currently indexed.

Public API
----------
VectorStore
    .add_chunks(chunks)
    .query(query_embedding, top_k, threshold) -> list[SearchResult]
    .delete_document(source_path)
    .list_documents() -> list[str]
    .count() -> int
    .reset()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

import chromadb
from chromadb.config import Settings

from rag_app.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    TOP_K_RESULTS,
    SIMILARITY_THRESHOLD,
)
from rag_app.logger import get_logger

log = get_logger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A single piece of text ready to be stored in the vector store."""
    chunk_id: str          # globally unique ID  e.g.  "paper_abc__0"
    text: str              # the chunk text
    embedding: List[float] # pre-computed embedding vector
    source: str            # original file path / URL
    page: int = 0          # page number (0 if unknown)
    section: str = ""      # section heading (optional)
    metadata: dict = field(default_factory=dict)  # any extra key-value pairs


@dataclass
class SearchResult:
    """One retrieved chunk, enriched with its similarity score."""
    chunk_id: str
    text: str
    source: str
    page: int
    section: str
    score: float           # cosine similarity  (0 – 1, higher = more relevant)
    metadata: dict = field(default_factory=dict)


# ── VectorStore ───────────────────────────────────────────────────────────────

class VectorStore:
    """
    Thin wrapper around a ChromaDB persistent collection.

    ChromaDB stores the collection on *CHROMA_DIR* so data survives between
    Python sessions.
    """

    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )
        log.info(
            "VectorStore ready — collection '%s', %d chunks on disk",
            COLLECTION_NAME,
            self._col.count(),
        )

    # ── Write ────────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Upsert *chunks* into the collection.

        Upserting (instead of inserting) means re-ingesting the same document
        is safe — existing chunks are simply overwritten.
        """
        if not chunks:
            log.warning("add_chunks called with empty list — nothing added.")
            return

        ids:        List[str]         = []
        embeddings: List[List[float]] = []
        documents:  List[str]         = []
        metadatas:  List[dict]        = []

        for c in chunks:
            ids.append(c.chunk_id)
            embeddings.append(c.embedding)
            documents.append(c.text)
            meta: dict[str, Any] = {
                "source":  c.source,
                "page":    c.page,
                "section": c.section,
            }
            meta.update(c.metadata)
            metadatas.append(meta)

        self._col.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        log.info("Upserted %d chunks into '%s'.", len(chunks), COLLECTION_NAME)

    # ── Read ─────────────────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: List[float],
        top_k: int = TOP_K_RESULTS,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> List[SearchResult]:
        """
        Return up to *top_k* chunks whose cosine similarity ≥ *threshold*.

        ChromaDB returns distances (0 = identical, 2 = opposite for cosine).
        We convert: similarity = 1 - distance.
        """
        if self._col.count() == 0:
            log.warning("Vector store is empty — no results returned.")
            return []

        raw = self._col.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._col.count()),
            include=["documents", "metadatas", "distances"],
        )

        results: List[SearchResult] = []
        for doc, meta, dist in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            score = round(1.0 - dist, 4)   # cosine similarity
            if score < threshold:
                continue
            results.append(
                SearchResult(
                    chunk_id=meta.get("chunk_id", ""),
                    text=doc,
                    source=meta.get("source", ""),
                    page=int(meta.get("page", 0)),
                    section=meta.get("section", ""),
                    score=score,
                    metadata={
                        k: v for k, v in meta.items()
                        if k not in {"source", "page", "section"}
                    },
                )
            )

        log.info(
            "Query returned %d/%d results above threshold %.2f",
            len(results), top_k, threshold,
        )
        return results

    # ── Delete ───────────────────────────────────────────────────────────────

    def delete_document(self, source_path: str) -> None:
        """Remove every chunk that came from *source_path*."""
        self._col.delete(where={"source": source_path})
        log.info("Deleted all chunks with source='%s'.", source_path)

    # ── Metadata helpers ─────────────────────────────────────────────────────

    def list_documents(self) -> List[str]:
        """Return sorted list of unique source paths stored in the collection."""
        if self._col.count() == 0:
            return []
        all_meta = self._col.get(include=["metadatas"])["metadatas"]
        return sorted({m.get("source", "") for m in all_meta if m})

    def count(self) -> int:
        """Total number of chunks currently in the collection."""
        return self._col.count()

    def reset(self) -> None:
        """Delete the entire collection and recreate it (destructive!)."""
        self._client.delete_collection(COLLECTION_NAME)
        self._col = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        log.warning("Collection '%s' has been reset.", COLLECTION_NAME)
