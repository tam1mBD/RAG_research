"""
tests/test_llm_client.py — Unit tests for LLMClient grounding validation.
Run with: pytest tests/test_llm_client.py -v
"""

import pytest
from unittest.mock import MagicMock, patch

from rag_app.llm_client import LLMClient, LLMResponse, _validate_grounding


# ── _validate_grounding (pure function — no mocks needed) ─────────────────────

def test_refusal_detected():
    answer = "I cannot answer this question based on the provided documents."
    result = _validate_grounding(answer, [])
    assert result["is_refused"] is True
    assert result["is_grounded"] is True
    assert result["warnings"] == []


def test_refusal_case_insensitive():
    answer = "i cannot answer this question based on the provided documents."
    result = _validate_grounding(answer, [])
    assert result["is_refused"] is True


def test_citation_extracted():
    answer = "Attention is key [Doc 1, p.3]."
    result = _validate_grounding(answer, ["paper.pdf"])
    assert not result["is_refused"]
    assert result["citations_found"] == ["[Doc 1"]
    assert result["is_grounded"] is True


def test_long_answer_no_citation_flags_ungrounded():
    long_answer = "word " * 40          # 40 words, no citation
    result = _validate_grounding(long_answer, ["paper.pdf"])
    assert result["is_grounded"] is False
    assert len(result["warnings"]) > 0


def test_short_answer_no_citation_warns_but_not_ungrounded():
    short_answer = "The model is accurate."
    result = _validate_grounding(short_answer, ["paper.pdf"])
    # short — should warn but not mark as ungrounded
    assert len(result["warnings"]) > 0


def test_out_of_range_citation_flagged():
    answer = "See [Doc 9, p.1]."
    result = _validate_grounding(answer, ["a.pdf", "b.pdf"])  # only 2 sources
    assert result["is_grounded"] is False
    assert any("out" in w.lower() or "range" in w.lower() for w in result["warnings"])


# ── LLMClient.generate (mocked LangChain) ─────────────────────────────────────

@pytest.fixture
def mock_llm_client(monkeypatch):
    with patch("rag_app.llm_client.ChatOpenAI") as MockChat:
        instance = MockChat.return_value
        ai_msg = MagicMock()
        ai_msg.content = "Self-attention captures dependencies [Doc 1, p.3]."
        ai_msg.usage_metadata = {
            "input_tokens": 120,
            "output_tokens": 30,
            "total_tokens": 150,
        }
        instance.invoke.return_value = ai_msg
        yield LLMClient()


def test_generate_returns_llm_response(mock_llm_client):
    messages = [{"role": "user", "content": "What is attention?"}]
    resp = mock_llm_client.generate(messages, context_sources=["paper.pdf"])
    assert isinstance(resp, LLMResponse)
    assert "attention" in resp.answer.lower()


def test_generate_populated_citations(mock_llm_client):
    messages = [{"role": "user", "content": "test"}]
    resp = mock_llm_client.generate(messages, context_sources=["paper.pdf"])
    assert resp.citations_found == ["[Doc 1"]


def test_generate_token_usage(mock_llm_client):
    messages = [{"role": "user", "content": "test"}]
    resp = mock_llm_client.generate(messages, context_sources=[])
    assert resp.prompt_tokens == 120
    assert resp.completion_tokens == 30
    assert resp.total_tokens == 150


def test_generate_refusal(monkeypatch):
    with patch("rag_app.llm_client.ChatOpenAI") as MockChat:
        instance = MockChat.return_value
        ai_msg = MagicMock()
        ai_msg.content = (
            "I cannot answer this question based on the provided documents."
        )
        ai_msg.usage_metadata = {}
        instance.invoke.return_value = ai_msg
        client = LLMClient()
        resp = client.generate([{"role": "user", "content": "capital?"}], [])
        assert resp.is_refused is True
        assert resp.is_grounded is True
