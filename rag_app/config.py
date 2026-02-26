"""
config.py — Centralised configuration loaded from environment variables.
All application modules import settings from here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present (ignored when env vars already set, e.g. in CI)
load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR        = Path(os.getenv("DOCS_DIR",        BASE_DIR / "data" / "docs"))
CHROMA_DIR      = Path(os.getenv("CHROMA_PERSIST_DIR", BASE_DIR / "data" / "chroma_db"))
CACHE_DIR       = Path(os.getenv("CACHE_DIR",        BASE_DIR / "data" / "embedding_cache"))

for _d in (DOCS_DIR, CHROMA_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── API / Models ──────────────────────────────────────────────────────────────
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL        = os.getenv("LLM_MODEL",       "gpt-4o-mini")

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_RESULTS         = int(os.getenv("TOP_K_RESULTS",         "5"))
SIMILARITY_THRESHOLD  = float(os.getenv("SIMILARITY_THRESHOLD","0.30"))

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# ── Token budgets ─────────────────────────────────────────────────────────────
MAX_PROMPT_TOKENS   = int(os.getenv("MAX_PROMPT_TOKENS",   "3500"))
MAX_RESPONSE_TOKENS = int(os.getenv("MAX_RESPONSE_TOKENS", "600"))

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ── ChromaDB collection name ──────────────────────────────────────────────────
COLLECTION_NAME = "research_papers"
