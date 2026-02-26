"""
tests/test_citation_mapper.py — Unit tests for the citation mapper.
Run with: pytest tests/test_citation_mapper.py -v
"""

import pytest
from rag_app.citation_mapper import CitationMapper, Citation
from rag_app.vector_store import SearchResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_result(
    source: str = "/docs/paper.pdf",
    page: int = 3,
    section: str = "Introduction",
    text: str = "Transformers rely on self-attention mechanisms.",
    score: float = 0.91,
) -> SearchResult:
    return SearchResult(
        chunk_id="c1",
        text=text,
        source=source,
        page=page,
        section=section,
        score=score,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_map_single_citation():
    mapper = CitationMapper()
    answer = "Self-attention is key [Doc 1, p.3]."
    results = [_make_result()]
    mr = mapper.map(answer, results)

    assert len(mr.citations) == 1
    c = mr.citations[0]
    assert c.doc_num == 1
    assert c.page == 3
    assert c.file_name == "paper.pdf"
    assert c.valid is True
    assert c.score == 0.91


def test_map_annotated_answer_contains_filename():
    mapper = CitationMapper()
    answer = "Key finding [Doc 1, p.3]."
    results = [_make_result()]
    mr = mapper.map(answer, results)
    assert "paper.pdf" in mr.annotated_answer


def test_map_no_citations():
    mapper = CitationMapper()
    answer = "No references here at all."
    mr = mapper.map(answer, [_make_result()])
    assert mr.citations == []
    assert mr.annotated_answer == answer


def test_map_out_of_range_citation():
    mapper = CitationMapper()
    answer = "Something [Doc 5, p.1]."           # only 1 result provided
    mr = mapper.map(answer, [_make_result()])
    assert 5 in mr.unresolved
    assert mr.citations[0].valid is False


def test_map_deduplicated_citations():
    mapper = CitationMapper()
    answer = "First [Doc 1, p.2]. Also [Doc 1, p.2] again."
    results = [_make_result()]
    mr = mapper.map(answer, results)
    assert len(mr.citations) == 1


def test_map_multiple_docs():
    mapper = CitationMapper()
    answer = "See [Doc 1, p.1] and [Doc 2, p.5]."
    r1 = _make_result(source="/docs/a.pdf", page=1)
    r2 = _make_result(source="/docs/b.pdf", page=5)
    mr = mapper.map(answer, [r1, r2])
    assert len(mr.citations) == 2
    assert mr.citations[0].file_name == "a.pdf"
    assert mr.citations[1].file_name == "b.pdf"


def test_format_bibliography_non_empty():
    mapper = CitationMapper()
    answer = "Result [Doc 1, p.3]."
    mr = mapper.map(answer, [_make_result()])
    bib = CitationMapper.format_bibliography(mr.citations)
    assert "References" in bib
    assert "paper.pdf" in bib
    assert "0.91" in bib


def test_format_bibliography_empty():
    bib = CitationMapper.format_bibliography([])
    assert bib == ""
