"""
scripts/ingest.py — CLI entry point for document ingestion.

Usage
-----
# Ingest everything in data/docs/
python scripts/ingest.py

# Ingest a specific file
python scripts/ingest.py --file data/docs/paper.pdf

# Ingest a custom directory
python scripts/ingest.py --dir path/to/folder

# Wipe the collection and re-ingest from scratch
python scripts/ingest.py --reset
"""

import argparse
import sys
from pathlib import Path

# Make sure project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_app.ingestion import ingest_file, ingest_directory
from rag_app.vector_store import VectorStore
from rag_app.config import DOCS_DIR
from rag_app.logger import get_logger

log = get_logger("ingest_cli")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest research documents into the RAG vector store."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--file", "-f",
        metavar="PATH",
        help="Path to a single PDF / TXT / MD file to ingest.",
    )
    group.add_argument(
        "--dir", "-d",
        metavar="DIR",
        default=str(DOCS_DIR),
        help=f"Directory to scan recursively (default: {DOCS_DIR}).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing vector store collection before ingesting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = VectorStore()

    if args.reset:
        log.warning("--reset flag set: wiping the existing collection.")
        store.reset()

    if args.file:
        total = ingest_file(args.file, store=store)
    else:
        total = ingest_directory(args.dir, store=store)

    log.info(
        "Done. %d chunk(s) now in the vector store "
        "(total collection size: %d).",
        total,
        store.count(),
    )
    # Print indexed documents for quick verification
    docs = store.list_documents()
    if docs:
        log.info("Indexed documents:")
        for d in docs:
            log.info("  • %s", Path(d).name)


if __name__ == "__main__":
    main()
