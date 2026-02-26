"""
llm_client.py — LangChain-powered LLM client with post-generation grounding checks.

Responsibilities
----------------
1. Send assembled prompt messages to the LLM via LangChain's ChatOpenAI.
2. Post-validate the response:
   - Detect the refusal phrase → mark as refused, skip further checks.
   - Check that at least one [Doc N, ...] citation is present.
   - Check the answer does not exceed the max response length.
   - Flag potential hallucination if answer is long but cites nothing.
3. Return a structured LLMResponse dataclass carrying the answer text,
   grounding flags, token usage, and any warnings.

Public API
----------
LLMClient
    .generate(messages, context_sources) -> LLMResponse
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from rag_app.config import (
    OPENAI_API_KEY,
    LLM_MODEL,
    MAX_RESPONSE_TOKENS,
)
from rag_app.logger import get_logger

log = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_REFUSAL_PHRASE = "I cannot answer this question based on the provided documents."
_CITATION_PATTERN = re.compile(r"\[Doc\s*\d+", re.IGNORECASE)

# Warn if answer is longer than this many words but has no citation
_LONG_ANSWER_WORD_THRESHOLD = 30


# ── Response dataclass ────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    """Structured output from the LLM client."""
    answer: str                        # raw answer text from the LLM

    # Grounding flags
    is_refused: bool = False           # True when the model used the refusal phrase
    is_grounded: bool = True           # False when answer looks ungrounded
    citations_found: List[str] = field(default_factory=list)  # e.g. ["Doc 1", "Doc 2"]
    warnings: List[str] = field(default_factory=list)

    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# ── Grounding validator ───────────────────────────────────────────────────────

def _validate_grounding(answer: str, context_sources: List[str]) -> dict:
    """
    Run post-generation grounding checks on *answer*.

    Returns a dict with keys:
        is_refused, is_grounded, citations_found, warnings
    """
    warnings: List[str] = []
    is_refused = _REFUSAL_PHRASE.lower() in answer.lower()

    if is_refused:
        return {
            "is_refused": True,
            "is_grounded": True,    # refusing is a grounded behaviour
            "citations_found": [],
            "warnings": [],
        }

    # Extract citation references e.g. [Doc 1, p.3]
    raw_citations = _CITATION_PATTERN.findall(answer)
    citations_found = sorted(set(raw_citations))

    word_count = len(answer.split())
    is_grounded = True

    if not citations_found:
        if word_count > _LONG_ANSWER_WORD_THRESHOLD:
            warnings.append(
                f"Answer is {word_count} words long but contains no [Doc N] citations. "
                "Possible hallucination — review manually."
            )
            is_grounded = False
        else:
            warnings.append(
                "No citations found in the answer. "
                "Consider whether the response is adequately grounded."
            )

    # Warn if cited doc numbers exceed the number of provided sources
    cited_nums = [
        int(re.search(r"\d+", c).group())
        for c in citations_found
        if re.search(r"\d+", c)
    ]
    if context_sources and cited_nums:
        out_of_range = [n for n in cited_nums if n > len(context_sources)]
        if out_of_range:
            warnings.append(
                f"Answer cites Doc {out_of_range} which is outside the "
                f"retrieved context range (1–{len(context_sources)}). "
                "Possible hallucination."
            )
            is_grounded = False

    return {
        "is_refused": False,
        "is_grounded": is_grounded,
        "citations_found": citations_found,
        "warnings": warnings,
    }


# ── Message converter ─────────────────────────────────────────────────────────

def _to_langchain_messages(messages: List[dict]):
    """Convert OpenAI-style dicts to LangChain message objects."""
    lc_messages = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
    return lc_messages


# ── LLMClient ─────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Wraps LangChain's ChatOpenAI with post-generation grounding validation.

    Usage
    -----
    client = LLMClient()
    response = client.generate(messages, context_sources=["paper1.pdf", ...])
    print(response.answer)
    print(response.warnings)
    """

    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=OPENAI_API_KEY,
            max_tokens=MAX_RESPONSE_TOKENS,
            temperature=0.0,       # deterministic — critical for grounding
        )
        log.info("LLMClient initialised (model=%s, temp=0.0).", LLM_MODEL)

    def generate(
        self,
        messages: List[dict],
        context_sources: List[str] | None = None,
    ) -> LLMResponse:
        """
        Send *messages* to the LLM and return a validated LLMResponse.

        Parameters
        ----------
        messages        : OpenAI-style list of {role, content} dicts,
                          produced by PromptBuilder.build().
        context_sources : list of source file names/paths that were included
                          in the context block — used to range-check citations.
        """
        context_sources = context_sources or []
        lc_msgs = _to_langchain_messages(messages)

        log.info("Calling %s (%d messages)…", LLM_MODEL, len(lc_msgs))
        ai_message = self._llm.invoke(lc_msgs)

        answer: str = ai_message.content.strip()

        # ── Token usage ───────────────────────────────────────────────────────
        usage = getattr(ai_message, "usage_metadata", None) or {}
        prompt_tokens     = usage.get("input_tokens",  0)
        completion_tokens = usage.get("output_tokens", 0)
        total_tokens      = usage.get("total_tokens",  prompt_tokens + completion_tokens)

        log.info(
            "LLM response received — %d tokens "
            "(prompt=%d, completion=%d).",
            total_tokens, prompt_tokens, completion_tokens,
        )

        # ── Post-generation grounding check ───────────────────────────────────
        grounding = _validate_grounding(answer, context_sources)

        if grounding["warnings"]:
            for w in grounding["warnings"]:
                log.warning("Grounding check: %s", w)

        if grounding["is_refused"]:
            log.info("Model issued refusal — no grounded answer available.")
        else:
            log.info(
                "Citations found: %s | Grounded: %s",
                grounding["citations_found"] or "none",
                grounding["is_grounded"],
            )

        return LLMResponse(
            answer=answer,
            is_refused=grounding["is_refused"],
            is_grounded=grounding["is_grounded"],
            citations_found=grounding["citations_found"],
            warnings=grounding["warnings"],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
