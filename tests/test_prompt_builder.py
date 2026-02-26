"""
tests/test_prompt_builder.py — Unit tests for the multi-layer prompt builder.
Run with: pytest tests/test_prompt_builder.py -v
"""

import pytest
from unittest.mock import patch

from rag_app.prompt_builder import PromptBuilder, _count_tokens
from rag_app.vector_store import SearchResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_result(text: str, doc_num: int = 1, score: float = 0.9) -> SearchResult:
    return SearchResult(
        chunk_id=f"chunk_{doc_num}",
        text=text,
        source=f"/docs/paper{doc_num}.pdf",
        page=doc_num,
        section="Introduction",
        score=score,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_build_returns_list_of_dicts():
    pb = PromptBuilder()
    results = [_make_result("Transformers use self-attention.")]
    messages = pb.build("What is self-attention?", results)
    assert isinstance(messages, list)
    assert all(isinstance(m, dict) for m in messages)
    assert all("role" in m and "content" in m for m in messages)


def test_build_has_five_messages():
    pb = PromptBuilder()
    messages = pb.build("test query", [_make_result("some text")])
    assert len(messages) == 5


def test_last_message_is_user_role():
    pb = PromptBuilder()
    messages = pb.build("my question", [_make_result("context")])
    assert messages[-1]["role"] == "user"
    assert "my question" in messages[-1]["content"]


def test_context_block_contains_doc_label():
    pb = PromptBuilder()
    results = [_make_result("Neural networks learn representations.")]
    messages = pb.build("Tell me about neural networks", results)
    context_msg = messages[3]["content"]
    assert "Doc 1" in context_msg
    assert "paper1.pdf" in context_msg


def test_empty_results_still_builds():
    pb = PromptBuilder()
    messages = pb.build("What is BERT?", [])
    assert len(messages) == 5
    assert "No relevant" in messages[3]["content"]


def test_token_budget_respected():
    """With a very tight budget, chunks should be trimmed."""
    pb = PromptBuilder()
    # 100 large chunks — only a subset should fit
    big_results = [_make_result("word " * 200, i) for i in range(100)]

    with patch("rag_app.prompt_builder.MAX_PROMPT_TOKENS", 1000), \
         patch("rag_app.prompt_builder.MAX_RESPONSE_TOKENS", 100):
        messages = pb.build("question", big_results)

    context_content = messages[3]["content"]
    # Should not contain all 100 doc labels
    assert context_content.count("Doc ") < 100


def test_token_count_positive():
    pb = PromptBuilder()
    msgs = pb.build("hello", [_make_result("world")])
    assert pb.token_count(msgs) > 0
