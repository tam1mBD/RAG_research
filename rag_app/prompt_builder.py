"""
prompt_builder.py — Structured multi-layer prompt assembly for the RAG Q&A assistant.

Five layers assembled in order
───────────────────────────────
Layer 1  SYSTEM ROLE        Who the model is and its core constraints.
Layer 2  TASK INSTRUCTIONS  Step-by-step reasoning rules + grounding mandate.
Layer 3  FEW-SHOT EXAMPLES  In-context demonstrations of correct behaviour.
Layer 4  CONTEXT BLOCK      Retrieved document chunks injected here.
Layer 5  USER QUERY         The user's question + output format constraints.

Token budget
────────────
`build()` respects MAX_PROMPT_TOKENS: if the context block would overflow
the budget, chunks are trimmed from the bottom (lowest similarity first,
they're already sorted descending by score).

Public API
──────────
PromptBuilder
    .build(query, results)  ->  list[dict]   (OpenAI messages format)
    .token_count(messages)  ->  int
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import tiktoken

from rag_app.config import MAX_PROMPT_TOKENS, MAX_RESPONSE_TOKENS, LLM_MODEL
from rag_app.vector_store import SearchResult
from rag_app.logger import get_logger

log = get_logger(__name__)


# ── Tokeniser ─────────────────────────────────────────────────────────────────

def _get_encoder():
    """Return the tiktoken encoder for the active LLM model."""
    try:
        return tiktoken.encoding_for_model(LLM_MODEL)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")   # safe fallback


_ENC = _get_encoder()


def _count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def _count_messages_tokens(messages: List[dict]) -> int:
    """Approximate token count for a list of OpenAI chat messages."""
    total = 0
    for m in messages:
        total += 4                               # per-message overhead
        total += _count_tokens(m.get("content", ""))
    total += 2                                   # reply primer
    return total


# ── Layer 1 — System Role ─────────────────────────────────────────────────────

_SYSTEM_ROLE = """\
You are a precise, citation-backed Research Paper Q&A Assistant.
Your sole knowledge source is the set of document excerpts provided in the \
CONTEXT block below.
You do NOT use any external knowledge, training data, or assumptions beyond \
what appears in those excerpts.\
"""


# ── Layer 2 — Task Instructions ───────────────────────────────────────────────

_TASK_INSTRUCTIONS = """\
## Task Instructions

Follow these rules exactly — no exceptions:

1. **Ground every claim** in the CONTEXT. If information is absent from the \
CONTEXT, state that clearly instead of guessing.
2. **Cite your sources** using [Doc N, p.X] notation after each statement \
that relies on a specific excerpt.
3. **Refusal rule**: If the question cannot be answered using the CONTEXT \
alone, respond with exactly:
   "I cannot answer this question based on the provided documents."
4. **No hallucination**: Do not invent facts, statistics, author names, or \
conclusions not present in the CONTEXT.
5. **Reasoning first**: Think step-by-step internally before writing the \
final answer —  but only output the final answer.
6. **Format**: Respond in clear, concise English. Use bullet points for lists \
and keep the answer under 5 sentences unless more detail is required.
7. **Conflicting excerpts**: If two excerpts contradict each other, acknowledge \
the conflict and cite both sources.\
"""


# ── Layer 3 — Few-Shot Examples ───────────────────────────────────────────────

_FEW_SHOT_EXAMPLES = """\
## Examples of Correct Behaviour

### Example 1 — Grounded answer with citation
CONTEXT excerpt:
  [Doc 1, p.3] "Transformer models rely on self-attention mechanisms to \
capture long-range dependencies in text."

Question: How do transformers capture long-range dependencies?
Answer: Transformer models use self-attention mechanisms to capture \
long-range dependencies in text [Doc 1, p.3].

---

### Example 2 — Out-of-context refusal
CONTEXT excerpt:
  [Doc 1, p.5] "The dataset contains 50,000 labelled sentences."

Question: What is the capital of France?
Answer: I cannot answer this question based on the provided documents.

---

### Example 3 — Conflicting sources
CONTEXT excerpt:
  [Doc 1, p.2] "The model achieved 92% accuracy on the test set."
  [Doc 2, p.8] "Accuracy on the same benchmark was reported as 87%."

Question: What accuracy did the model achieve?
Answer: The reported accuracy differs across sources: Doc 1 states 92% \
[Doc 1, p.2] while Doc 2 reports 87% [Doc 2, p.8]. The discrepancy may \
reflect different evaluation conditions.\
"""


# ── Layer 4 — Context Block builder ──────────────────────────────────────────

def _build_context_block(results: List[SearchResult]) -> tuple[str, List[SearchResult]]:
    """
    Format retrieved chunks into a numbered CONTEXT block.

    Returns
    -------
    (context_text, included_results)
    included_results may be a subset of results if token trimming occurred.
    """
    if not results:
        return "No relevant document excerpts were retrieved.", []

    lines: List[str] = ["## Context\n"]
    included: List[SearchResult] = []

    for i, r in enumerate(results, start=1):
        doc_label = f"Doc {i}"
        source_name = Path(r.source).name if r.source else "unknown"
        page_info = f"p.{r.page}" if r.page else "p.?"
        section_info = f" | {r.section}" if r.section else ""
        score_info = f"similarity={r.score:.2f}"

        header = (
            f"[{doc_label}, {page_info}]  "
            f"Source: {source_name}{section_info}  ({score_info})"
        )
        lines.append(header)
        lines.append(r.text.strip())
        lines.append("")          # blank line between chunks
        included.append(r)

    return "\n".join(lines), included


# ── Layer 5 — Query + Format constraint ──────────────────────────────────────

def _build_user_turn(query: str) -> str:
    return (
        f"## Question\n{query.strip()}\n\n"
        "## Answer\n"
        "Respond in ≤5 sentences. Cite every factual claim with [Doc N, p.X]. "
        "If the answer is not in the context, use the refusal phrase exactly."
    )


# ── PromptBuilder ─────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Assembles the full 5-layer prompt and enforces the token budget.

    Token budget
    ────────────
    Available tokens for context =
        MAX_PROMPT_TOKENS - MAX_RESPONSE_TOKENS - fixed_layer_tokens

    If retrieved chunks exceed that budget, they are trimmed from the end
    (lowest-relevance chunks removed first, since results arrive sorted
    descending by similarity score).
    """

    def __init__(self) -> None:
        # Pre-compute token cost of the fixed layers (1-3 + 5 skeleton)
        self._fixed_messages = [
            {"role": "system",    "content": _SYSTEM_ROLE},
            {"role": "assistant", "content": _TASK_INSTRUCTIONS},
            {"role": "assistant", "content": _FEW_SHOT_EXAMPLES},
        ]
        self._fixed_tokens = _count_messages_tokens(self._fixed_messages)
        log.debug("Fixed layers token cost: %d", self._fixed_tokens)

    # ── Public ────────────────────────────────────────────────────────────────

    def build(
        self,
        query: str,
        results: List[SearchResult],
    ) -> List[dict]:
        """
        Assemble the complete prompt as an OpenAI-compatible messages list.

        Parameters
        ----------
        query   : the user's raw question string.
        results : ranked list of SearchResult objects (highest score first).

        Returns
        -------
        list[dict]  e.g. [{"role": "system", "content": "..."},  ...]
        """
        user_turn_tokens = _count_tokens(_build_user_turn(query))
        context_budget = (
            MAX_PROMPT_TOKENS
            - MAX_RESPONSE_TOKENS
            - self._fixed_tokens
            - user_turn_tokens
            - 20          # safety buffer
        )

        # ── Trim chunks to fit budget ─────────────────────────────────────────
        trimmed_results: List[SearchResult] = []
        running_tokens = 0
        for r in results:                       # already sorted best→worst
            chunk_tokens = _count_tokens(r.text)
            if running_tokens + chunk_tokens > context_budget:
                log.info(
                    "Token budget reached — dropping %d lower-relevance chunk(s).",
                    len(results) - len(trimmed_results),
                )
                break
            trimmed_results.append(r)
            running_tokens += chunk_tokens

        context_block, included = _build_context_block(trimmed_results)

        messages = [
            # Layer 1 — System role
            {
                "role": "system",
                "content": _SYSTEM_ROLE,
            },
            # Layer 2 — Task instructions
            {
                "role": "system",
                "content": _TASK_INSTRUCTIONS,
            },
            # Layer 3 — Few-shot examples
            {
                "role": "system",
                "content": _FEW_SHOT_EXAMPLES,
            },
            # Layer 4 — Retrieved context
            {
                "role": "system",
                "content": context_block,
            },
            # Layer 5 — User query + format constraint
            {
                "role": "user",
                "content": _build_user_turn(query),
            },
        ]

        total_tokens = _count_messages_tokens(messages)
        log.info(
            "Prompt built — %d chunk(s) included, ~%d total tokens "
            "(budget %d).",
            len(included),
            total_tokens,
            MAX_PROMPT_TOKENS,
        )
        return messages

    def token_count(self, messages: List[dict]) -> int:
        """Return approximate token count for *messages*."""
        return _count_messages_tokens(messages)
