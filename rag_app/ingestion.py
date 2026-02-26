"""
ingestion.py — Document loading and chunking pipeline using LangChain.

Flow
----
file (PDF / TXT / MD)
  └─► LangChain Loader          (PyPDFLoader / TextLoader / UnstructuredMarkdownLoader)
        └─► LangChain Splitter  (RecursiveCharacterTextSplitter)
              └─► embed_batch() (our cached embedder)
                    └─► VectorStore.add_chunks()

Public API
----------
ingest_file(path)     -> int   (number of chunks added)
ingest_directory(dir) -> int   (total chunks added)
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import List

# ── LangChain ─────────────────────────────────────────────────────────────────
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ── Internal ──────────────────────────────────────────────────────────────────
from rag_app.config import CHUNK_SIZE, CHUNK_OVERLAP, DOCS_DIR
from rag_app.embedder import embed_batch
from rag_app.vector_store import Chunk, VectorStore
from rag_app.logger import get_logger

log = get_logger(__name__)

# ── Splitter (shared instance) ────────────────────────────────────────────────
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)


# ── Loader selection ──────────────────────────────────────────────────────────

_LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md":  UnstructuredMarkdownLoader,
}


def _get_loader(path: Path):
    suffix = path.suffix.lower()
    loader_cls = _LOADER_MAP.get(suffix)
    if loader_cls is None:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported: {list(_LOADER_MAP.keys())}"
        )
    if loader_cls is TextLoader:
        return loader_cls(str(path), encoding="utf-8", autodetect_encoding=True)
    return loader_cls(str(path))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_chunk_id(source: str, index: int) -> str:
    """Stable, unique ID for a chunk derived from the source path + index."""
    digest = hashlib.md5(source.encode()).hexdigest()[:8]
    return f"{digest}__{index}"


def _extract_section(text: str) -> str:
    """
    Heuristic: return the first line that looks like a section heading,
    or empty string if none found.
    """
    for line in text.splitlines():
        line = line.strip()
        # Markdown heading
        if re.match(r"^#{1,4}\s+\S", line):
            return re.sub(r"^#+\s+", "", line)
        # ALL-CAPS or Title Case short line (≤ 80 chars) — likely a heading
        if len(line) <= 80 and (line.isupper() or re.match(r"^[A-Z][a-z]", line)):
            return line
    return ""


# ── Core functions ────────────────────────────────────────────────────────────

def ingest_file(path: str | Path, store: VectorStore | None = None) -> int:
    """
    Load *path*, split into chunks, embed and upsert into *store*.

    Parameters
    ----------
    path  : path to a PDF, TXT, or MD file.
    store : existing VectorStore instance; created fresh if None.

    Returns
    -------
    Number of chunks successfully stored.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    if store is None:
        store = VectorStore()

    log.info("Loading '%s'…", path.name)
    loader = _get_loader(path)
    lc_docs = loader.load()           # list[langchain_core.documents.Document]

    if not lc_docs:
        log.warning("Loader returned 0 documents for '%s'. Skipping.", path.name)
        return 0

    # Split every loaded page / section into smaller chunks
    split_docs = _splitter.split_documents(lc_docs)
    log.info("  → %d raw chunks after splitting.", len(split_docs))

    if not split_docs:
        return 0

    # Build texts list for batch embedding
    texts = [doc.page_content for doc in split_docs]

    log.info("  → Embedding %d chunks (cached embedder)…", len(texts))
    embeddings = embed_batch(texts)

    source_str = str(path)
    chunks: List[Chunk] = []
    for idx, (doc, emb) in enumerate(zip(split_docs, embeddings)):
        page = int(doc.metadata.get("page", 0))
        chunks.append(
            Chunk(
                chunk_id=_make_chunk_id(source_str, idx),
                text=doc.page_content,
                embedding=emb,
                source=source_str,
                page=page,
                section=_extract_section(doc.page_content),
                metadata={"file_name": path.name},
            )
        )

    store.add_chunks(chunks)
    log.info("  → Stored %d chunks from '%s'.", len(chunks), path.name)
    return len(chunks)


def ingest_directory(
    directory: str | Path = DOCS_DIR,
    store: VectorStore | None = None,
) -> int:
    """
    Recursively ingest all supported files inside *directory*.

    Returns total number of chunks stored across all files.
    """
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    if store is None:
        store = VectorStore()

    supported = list(_LOADER_MAP.keys())
    files = [
        f for f in directory.rglob("*")
        if f.is_file() and f.suffix.lower() in supported
    ]

    if not files:
        log.warning(
            "No supported files found in '%s'. "
            "Drop PDF / TXT / MD files there and try again.",
            directory,
        )
        return 0

    log.info("Found %d file(s) to ingest in '%s'.", len(files), directory)
    total = 0
    for f in files:
        try:
            total += ingest_file(f, store=store)
        except Exception as exc:
            log.error("Failed to ingest '%s': %s", f.name, exc)

    log.info("Ingestion complete — %d total chunks stored.", total)
    return total
