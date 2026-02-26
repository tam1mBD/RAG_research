"""
scripts/ask.py — CLI interface for the RAG pipeline.

Usage
-----
python scripts/ask.py "What is the attention mechanism?"
python scripts/ask.py "Summarise the results" --top-k 8 --threshold 0.4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_app.pipeline import RAGPipeline
from rag_app.logger import get_logger

log = get_logger("ask_cli")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask a question against your ingested research documents."
    )
    parser.add_argument("query", help="Your question (wrap in quotes).")
    parser.add_argument(
        "--top-k", "-k", type=int, default=None,
        help="Number of chunks to retrieve (default: from config).",
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=None,
        help="Minimum similarity score 0–1 (default: from config).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = RAGPipeline()

    kwargs: dict = {}
    if args.top_k is not None:
        kwargs["top_k"] = args.top_k
    if args.threshold is not None:
        kwargs["threshold"] = args.threshold

    response = pipeline.ask(args.query, **kwargs)
    print("\n" + str(response))


if __name__ == "__main__":
    main()
