"""
citation_mapper.py — Maps [Doc N, p.X] references in LLM answers back to
the exact source document, page, section, and text snippet.

Flow
----
LLM answer (contains "[Doc 1, p.3]", "[Doc 2, p.7]" …)
  + ordered list of SearchResult objects (same order as context block)
  ─►  CitationMapper.map(answer, results)
        ─►  list[Citation]          (one per unique Doc N reference)
        ─►  annotated_answer str    (inline refs enriched with file name)

Public API
----------
Citation           (dataclass)
CitationMapper
    .map(answer, results)   -> MappingResult
    .format_bibliography(citations) -> str
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from rag_app.vector_store import SearchResult
from rag_app.logger import get_logger

log = get_logger(__name__)

# Matches patterns like:
#   [Doc 1, p.3]   [Doc 2]   [Doc 3, p.?]   [doc 1, p.12]
_CITATION_RE = re.compile(
    r"\[Doc\s*(?P<num>\d+)(?:,\s*p\.(?P<page>[0-9?]+))?\]",
    re.IGNORECASE,
)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Citation:
    """One resolved citation reference."""
    doc_num: int           # as referenced in the answer  (1-based)
    source: str            # full path of the source file
    file_name: str         # just the filename  e.g. "attention.pdf"
    page: int              # page number from metadata (0 = unknown)
    cited_page: str        # page string as written in the answer ("3" / "?")
    section: str           # section heading if available
    snippet: str           # first 200 chars of the matched chunk text
    score: float           # retrieval similarity score
    valid: bool = True     # False when Doc N > len(results)


@dataclass
class MappingResult:
    """Output of CitationMapper.map()."""
    citations: List[Citation]          # deduplicated, sorted by doc_num
    annotated_answer: str              # answer with refs replaced by richer labels
    unresolved: List[int] = field(default_factory=list)  # doc_nums with no match


# ── CitationMapper ────────────────────────────────────────────────────────────

class CitationMapper:
    """
    Resolves [Doc N, p.X] references in an LLM answer against the ordered
    list of SearchResult objects that were placed in the context block.

    The doc number N in "[Doc N]" corresponds directly to the position of
    the result in the *results* list passed to PromptBuilder.build()
    (1-based index).
    """

    # Truncate snippet to this many characters
    SNIPPET_LEN = 220

    def map(
        self,
        answer: str,
        results: List[SearchResult],
    ) -> MappingResult:
        """
        Parameters
        ----------
        answer  : raw answer string returned by the LLM.
        results : SearchResult list in the same order they were given to the
                  prompt builder (index 0 → Doc 1, index 1 → Doc 2, …).

        Returns
        -------
        MappingResult
        """
        matches = list(_CITATION_RE.finditer(answer))

        if not matches:
            log.info("No [Doc N] citations found in the answer.")
            return MappingResult(citations=[], annotated_answer=answer)

        seen_nums: set[int] = set()
        citations: List[Citation] = []
        unresolved: List[int] = []

        for m in matches:
            num = int(m.group("num"))
            if num in seen_nums:
                continue
            seen_nums.add(num)

            cited_page = m.group("page") or "?"
            idx = num - 1   # convert to 0-based

            if idx < 0 or idx >= len(results):
                log.warning(
                    "Answer references [Doc %d] but only %d result(s) "
                    "were retrieved. Marking as unresolved.",
                    num, len(results),
                )
                unresolved.append(num)
                citations.append(
                    Citation(
                        doc_num=num,
                        source="",
                        file_name="unknown",
                        page=0,
                        cited_page=cited_page,
                        section="",
                        snippet="⚠ Source not found — citation out of range.",
                        score=0.0,
                        valid=False,
                    )
                )
                continue

            r = results[idx]
            file_name = Path(r.source).name if r.source else "unknown"
            snippet = r.text.strip()[: self.SNIPPET_LEN]
            if len(r.text.strip()) > self.SNIPPET_LEN:
                snippet += "…"

            citations.append(
                Citation(
                    doc_num=num,
                    source=r.source,
                    file_name=file_name,
                    page=r.page,
                    cited_page=cited_page,
                    section=r.section,
                    snippet=snippet,
                    score=r.score,
                    valid=True,
                )
            )
            log.debug(
                "Resolved [Doc %d] → '%s' p.%s (score=%.2f)",
                num, file_name, cited_page, r.score,
            )

        citations.sort(key=lambda c: c.doc_num)

        # ── Annotate answer: replace [Doc N, p.X] with richer inline label ───
        def _replacer(m: re.Match) -> str:
            num = int(m.group("num"))
            cited_page = m.group("page") or "?"
            idx = num - 1
            if 0 <= idx < len(results):
                fname = Path(results[idx].source).name if results[idx].source else "?"
                return f"[Doc {num}, p.{cited_page} · {fname}]"
            return m.group(0)   # leave unchanged if unresolved

        annotated = _CITATION_RE.sub(_replacer, answer)

        log.info(
            "Citation mapping complete — %d unique ref(s), %d unresolved.",
            len(citations), len(unresolved),
        )

        return MappingResult(
            citations=citations,
            annotated_answer=annotated,
            unresolved=unresolved,
        )

    # ── Bibliography formatter ────────────────────────────────────────────────

    @staticmethod
    def format_bibliography(citations: List[Citation]) -> str:
        """
        Return a formatted bibliography block for display in the UI.

        Example output
        --------------
        References
        ──────────
        [Doc 1]  attention_is_all_you_need.pdf  |  p.3  |  score 0.91
                 Section: Introduction
                 "We propose a new simple network architecture, the Transformer…"

        [Doc 2]  bert_paper.pdf  |  p.7  |  score 0.84
                 ⚠ Citation out of range — source unavailable.
        """
        if not citations:
            return ""

        lines = ["References", "─" * 60]
        for c in citations:
            status = "⚠ Citation out of range — source unavailable." if not c.valid else ""
            section_line = f"         Section : {c.section}" if c.section else ""
            snippet_line = f'         "{c.snippet}"' if c.valid else f"         {status}"

            lines.append(
                f"[Doc {c.doc_num}]  {c.file_name}  |  "
                f"p.{c.cited_page}  |  score {c.score:.2f}"
            )
            if section_line:
                lines.append(section_line)
            lines.append(snippet_line)
            lines.append("")

        return "\n".join(lines).rstrip()
