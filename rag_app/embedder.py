"""
embedder.py — Embedding client with disk-based SHA-256 cache.

Every unique text string is hashed; if its embedding already exists on disk it
is returned immediately, skipping the API call and reducing latency + cost.

Public API
----------
embed_text(text: str) -> list[float]
embed_batch(texts: list[str]) -> list[list[float]]
"""

import hashlib
import json
from pathlib import Path
from typing import List

import openai

from rag_app.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    CACHE_DIR,
)
from rag_app.logger import get_logger

log = get_logger(__name__)

# Initialise the OpenAI client once
_client = openai.OpenAI(api_key=OPENAI_API_KEY)


# ── Cache helpers ─────────────────────────────────────────────────────────────

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


# ── Public functions ──────────────────────────────────────────────────────────

def embed_text(text: str) -> List[float]:
    """
    Return the embedding vector for *text*.
    Uses the disk cache; only calls the API on a cache miss.
    """
    text = text.strip()
    if not text:
        raise ValueError("Cannot embed an empty string.")

    cached = _load_from_cache(text)
    if cached is not None:
        log.debug("Cache hit for text (len=%d)", len(text))
        return cached

    log.debug("Cache miss — calling OpenAI Embeddings API (len=%d)", len(text))
    response = _client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
    )
    embedding: List[float] = response.data[0].embedding
    _save_to_cache(text, embedding)
    return embedding


def embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Return embedding vectors for a list of texts.

    Strategy
    --------
    1. Check cache for every item.
    2. Send all cache-miss items to the API in a *single* batched request.
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
        "embed_batch: %d cached, %d API calls needed",
        len(texts) - len(miss_texts),
        len(miss_texts),
    )

    if miss_texts:
        response = _client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=miss_texts,
        )
        # API returns results in the same order as input
        for api_idx, item in enumerate(response.data):
            original_idx = miss_indices[api_idx]
            embedding = item.embedding
            _save_to_cache(miss_texts[api_idx], embedding)
            results[original_idx] = embedding

    return results  # type: ignore[return-value]
