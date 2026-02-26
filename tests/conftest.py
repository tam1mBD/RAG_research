"""
tests/conftest.py — Shared pytest fixtures for the test suite.
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def _ensure_env(tmp_path_factory):
    """
    Set dummy env vars so config.py doesn't blow up during unit tests.
    The actual API is never called in unit tests (everything is mocked).
    """
    import os
    os.environ.setdefault("OPENAI_API_KEY", "sk-test-000")
    os.environ.setdefault("EMBEDDING_MODEL", "text-embedding-3-small")
    os.environ.setdefault("LLM_MODEL", "gpt-4o-mini")
