"""
tests/test_embedder.py — Unit tests for the embedding module.
Run with: pytest tests/test_embedder.py -v
"""

import json
import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────

FAKE_EMBEDDING = [0.1] * 1536      # text-embedding-3-small dimension


@pytest.fixture(autouse=True)
def _patch_openai(tmp_path, monkeypatch):
    """Redirect cache to a temp dir and mock the OpenAI client."""
    import rag_app.embedder as emb_mod

    monkeypatch.setattr(emb_mod, "CACHE_DIR", tmp_path)

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=FAKE_EMBEDDING)]
    )
    monkeypatch.setattr(emb_mod, "_client", mock_client)
    return mock_client


# ── embed_text ────────────────────────────────────────────────────────────────

def test_embed_text_returns_vector():
    from rag_app.embedder import embed_text
    result = embed_text("hello world")
    assert isinstance(result, list)
    assert len(result) == 1536


def test_embed_text_caches_on_disk(tmp_path, monkeypatch):
    import rag_app.embedder as emb_mod
    monkeypatch.setattr(emb_mod, "CACHE_DIR", tmp_path)

    from rag_app.embedder import embed_text
    text = "cache me"
    embed_text(text)

    digest = hashlib.sha256(text.encode()).hexdigest()
    cache_files = list(tmp_path.glob("*.json"))
    assert any(digest in f.name for f in cache_files), "Cache file not created"


def test_embed_text_cache_hit_skips_api(tmp_path, monkeypatch):
    import rag_app.embedder as emb_mod
    monkeypatch.setattr(emb_mod, "CACHE_DIR", tmp_path)

    from rag_app.embedder import embed_text, _cache_key, _save_to_cache
    text = "already cached"
    _save_to_cache(text, FAKE_EMBEDDING)

    embed_text(text)
    emb_mod._client.embeddings.create.assert_not_called()


def test_embed_text_empty_raises():
    from rag_app.embedder import embed_text
    with pytest.raises(ValueError):
        embed_text("   ")


# ── embed_batch ───────────────────────────────────────────────────────────────

def test_embed_batch_returns_correct_length():
    from rag_app.embedder import embed_batch
    texts = ["alpha", "beta", "gamma"]
    results = embed_batch(texts)
    assert len(results) == 3
    assert all(len(r) == 1536 for r in results)


def test_embed_batch_empty_list():
    from rag_app.embedder import embed_batch
    assert embed_batch([]) == []


def test_embed_batch_partial_cache(tmp_path, monkeypatch):
    import rag_app.embedder as emb_mod
    monkeypatch.setattr(emb_mod, "CACHE_DIR", tmp_path)

    from rag_app.embedder import embed_batch, _save_to_cache

    cached_text = "already here"
    _save_to_cache(cached_text, FAKE_EMBEDDING)

    # Only 1 of 2 texts needs an API call
    emb_mod._client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=FAKE_EMBEDDING)]
    )
    results = embed_batch([cached_text, "new text"])
    assert len(results) == 2
    emb_mod._client.embeddings.create.assert_called_once()
