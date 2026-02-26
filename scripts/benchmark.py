"""
scripts/benchmark.py — Accuracy & hallucination benchmark for the RAG pipeline.

Runs a set of benchmark questions against the indexed documents, compares
answers to expected keywords, and produces a summary report.

Usage
-----
python scripts/benchmark.py
python scripts/benchmark.py --questions scripts/benchmark_questions.json
python scripts/benchmark.py --output results/benchmark_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_app.pipeline import RAGPipeline, RAGResponse
from rag_app.logger import get_logger

log = get_logger("benchmark")


# ── Default benchmark questions ───────────────────────────────────────────────
# Each entry: question + expected_keywords (any match = pass) + expect_refusal flag

DEFAULT_QUESTIONS = [
    {
        "id": "Q1",
        "question": "What is the attention mechanism and how does it work?",
        "expected_keywords": ["attention", "query", "key", "value", "weight"],
        "expect_refusal": False,
        "category": "Comprehension",
    },
    {
        "id": "Q2",
        "question": "What datasets were used to evaluate the model?",
        "expected_keywords": ["dataset", "benchmark", "corpus", "train", "test"],
        "expect_refusal": False,
        "category": "Factual",
    },
    {
        "id": "Q3",
        "question": "What accuracy or F1-score did the model achieve?",
        "expected_keywords": ["accuracy", "f1", "score", "%", "performance"],
        "expect_refusal": False,
        "category": "Numerical",
    },
    {
        "id": "Q4",
        "question": "What are the main limitations of the proposed approach?",
        "expected_keywords": ["limitation", "drawback", "future", "constraint", "however"],
        "expect_refusal": False,
        "category": "Critical",
    },
    {
        "id": "Q5",
        "question": "What is the capital city of France?",
        "expected_keywords": [],
        "expect_refusal": True,
        "category": "Refusal",
    },
    {
        "id": "Q6",
        "question": "Who are the authors of the paper?",
        "expected_keywords": ["author", "et al", "university", "proposed", "present"],
        "expect_refusal": False,
        "category": "Factual",
    },
    {
        "id": "Q7",
        "question": "How does the model handle long sequences?",
        "expected_keywords": ["sequence", "length", "position", "memory", "token"],
        "expect_refusal": False,
        "category": "Comprehension",
    },
    {
        "id": "Q8",
        "question": "What future work do the authors suggest?",
        "expected_keywords": ["future", "work", "extend", "improve", "direction"],
        "expect_refusal": False,
        "category": "Synthesis",
    },
]


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class QuestionResult:
    id: str
    question: str
    category: str
    answer: str
    is_refused: bool
    is_grounded: bool
    expect_refusal: bool
    keyword_matched: bool
    correct_refusal: bool     # True when refusal expectation matches actual
    warnings: List[str]
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    passed: bool = False


@dataclass
class BenchmarkReport:
    total: int = 0
    passed: int = 0
    failed: int = 0
    refusals_correct: int = 0
    grounded_count: int = 0
    avg_latency: float = 0.0
    avg_tokens: float = 0.0
    results: List[QuestionResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return round(self.passed / self.total * 100, 1) if self.total else 0.0

    @property
    def grounding_rate(self) -> float:
        return round(self.grounded_count / self.total * 100, 1) if self.total else 0.0


# ── Runner ────────────────────────────────────────────────────────────────────

def run_benchmark(questions: list[dict], pipeline: RAGPipeline) -> BenchmarkReport:
    report = BenchmarkReport(total=len(questions))
    latencies: list[float] = []
    tokens: list[int] = []

    print(f"\n{'─'*70}")
    print(f"  Running {len(questions)} benchmark question(s)…")
    print(f"{'─'*70}\n")

    for q in questions:
        qid       = q["id"]
        question  = q["question"]
        keywords  = [k.lower() for k in q.get("expected_keywords", [])]
        exp_ref   = q.get("expect_refusal", False)
        category  = q.get("category", "General")

        log.info("[%s] %s", qid, question)

        resp: RAGResponse = pipeline.ask(question)

        answer_lower = resp.answer.lower()

        # ── Keyword match ─────────────────────────────────────────────────────
        keyword_matched = (
            any(kw in answer_lower for kw in keywords)
            if keywords else True          # no keywords = grading via refusal only
        )

        # ── Refusal correctness ───────────────────────────────────────────────
        correct_refusal = (resp.is_refused == exp_ref)

        # ── Pass / Fail ───────────────────────────────────────────────────────
        if exp_ref:
            passed = resp.is_refused          # refusal questions: must refuse
        else:
            passed = keyword_matched and resp.is_grounded and not resp.is_refused

        status = "✅ PASS" if passed else "❌ FAIL"
        print(
            f"[{qid}] {category:<14} {status}  "
            f"| refused={resp.is_refused}  grounded={resp.is_grounded}  "
            f"kw_match={keyword_matched}  {resp.latency_seconds:.2f}s"
        )

        qr = QuestionResult(
            id=qid,
            question=question,
            category=category,
            answer=resp.answer,
            is_refused=resp.is_refused,
            is_grounded=resp.is_grounded,
            expect_refusal=exp_ref,
            keyword_matched=keyword_matched,
            correct_refusal=correct_refusal,
            warnings=resp.warnings,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            latency_seconds=resp.latency_seconds,
            passed=passed,
        )
        report.results.append(qr)

        if passed:
            report.passed += 1
        else:
            report.failed += 1
        if resp.is_refused and exp_ref:
            report.refusals_correct += 1
        if resp.is_grounded:
            report.grounded_count += 1

        latencies.append(resp.latency_seconds)
        tokens.append(resp.total_tokens)

    report.avg_latency = round(sum(latencies) / len(latencies), 3) if latencies else 0.0
    report.avg_tokens  = round(sum(tokens) / len(tokens), 1) if tokens else 0.0
    return report


# ── Summary printer ───────────────────────────────────────────────────────────

def print_summary(report: BenchmarkReport) -> None:
    print(f"\n{'═'*70}")
    print("  BENCHMARK SUMMARY")
    print(f"{'═'*70}")
    print(f"  Total questions : {report.total}")
    print(f"  Passed          : {report.passed}  ({report.accuracy}%)")
    print(f"  Failed          : {report.failed}")
    print(f"  Grounding rate  : {report.grounding_rate}%")
    print(f"  Avg latency     : {report.avg_latency}s")
    print(f"  Avg tokens/call : {report.avg_tokens}")
    print(f"{'═'*70}\n")

    # Per-category breakdown
    categories: dict[str, dict] = {}
    for r in report.results:
        cat = r.category
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0}
        categories[cat]["total"] += 1
        if r.passed:
            categories[cat]["passed"] += 1

    print("  Per-category accuracy:")
    for cat, counts in sorted(categories.items()):
        pct = round(counts["passed"] / counts["total"] * 100)
        bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
        print(f"    {cat:<16} {bar}  {counts['passed']}/{counts['total']}  ({pct}%)")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run accuracy/hallucination benchmark on the RAG pipeline."
    )
    parser.add_argument(
        "--questions", "-q",
        metavar="JSON",
        help="Path to a JSON file with benchmark questions (uses built-in set if omitted).",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="JSON",
        help="Save the full report to this JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.questions:
        path = Path(args.questions)
        if not path.exists():
            log.error("Questions file not found: %s", path)
            sys.exit(1)
        with path.open() as f:
            questions = json.load(f)
    else:
        questions = DEFAULT_QUESTIONS

    pipeline = RAGPipeline()

    if pipeline.index_status()["total_chunks"] == 0:
        log.warning(
            "Vector store is empty! Ingest documents first:\n"
            "  python scripts/ingest.py"
        )
        sys.exit(1)

    report = run_benchmark(questions, pipeline)
    print_summary(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(asdict(report), f, indent=2)
        log.info("Report saved to %s", out_path)


if __name__ == "__main__":
    main()
