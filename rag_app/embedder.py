"""
embedder.py -- Local embedding client using sentence-transformers, with disk-based SHA-256 cache.

Groq does not provide an embeddings API, so embeddings are generated locally
using sentence-transformers (no API key or cost required).

Default model : all-MiniLM-L6-v2  (384-dim, fast, high quality)
Override via  : EMBEDDING_MODEL env var

Every unique text string is hashed; if its embedding already exists on disk it
is returned immediately, skipping the model call and reducing latency.

Public API
----------
embed_text(text: str)          -> list[float]
embed_batch(texts: list[str])  -> list[list[float]]
"""

import hashlib
import json
from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer

from rag_app.config import (
    EMBEDDING_MODEL,
    CACHE_DIR,
)
from rag_app.logger import get_logger

log = get_logger(__name__)

# Load the local model once at import time
_model = SentenceTransformer(EMBEDDING_MODEL)
log.info("Embedder ready -- model=%s", EMBEDDING_MODEL)


# -- Cache helpers -------------------------------------------------------------

def _cache_key(text: str) -> str:
    """Return a deterministic filename for *text*."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{EMBEDDING_MODEL}_{digest}.json"


def _load_from_cache(text: str) -> List[float] | None:
    path = CACHE_DIR / _cache_key(text)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_to_cache(text: str, embedding: List[float]) -> None:
    path = CACHE_DIR / _cache_key(text)
    with path.open("w", encoding="utf-8") as f:
        json.dump(embedding, f)


# -- Public functions ----------------------------------------------------------

def embed_text(text: str) -> List[float]:
    """
    Return the embedding vector for *text*.
    Uses the disk cache; only runs the local model on a cache miss.
    """
    text = text.strip()
    if not text:
        raise ValueError("Cannot embed an empty string.")

    cached = _load_from_cache(text)
    if cached is not None:
        log.debug("Cache hit for text (len=%d)", len(text))
        return cached

    log.debug("Cache miss -- encoding locally (len=%d)", len(text))
    embedding: List[float] = _model.encode([text])[0].tolist()
    _save_to_cache(text, embedding)
    return embedding


def embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Return embedding vectors for a list of texts.

    Strategy
    --------
    1. Check cache for every item.
    2. Encode all cache-miss items in a single local batch call.
    3. Merge cached and fresh results preserving original order.
    """
    texts = [t.strip() for t in texts]
    if not texts:
        return []

    results: List[List[float] | None] = [None] * len(texts)
    miss_indices: List[int] = []
    miss_texts: List[str] = []

    for i, text in enumerate(texts):
        cached = _load_from_cache(text)
        if cached is not None:
            results[i] = cached
        else:
            miss_indices.append(i)
            miss_texts.append(text)

    log.info(
        "embed_batch: %d cached, %d to encode locally",
        len(texts) - len(miss_texts),
        len(miss_texts),
    )

    if miss_texts:
        embeddings = _model.encode(miss_texts)   # returns numpy array (n, dim)
        for i, original_idx in enumerate(miss_indices):
            embedding_list: List[float] = embeddings[i].tolist()
            _save_to_cache(miss_texts[i], embedding_list)
            results[original_idx] = embedding_list

    return results  # type: ignore[return-value]
