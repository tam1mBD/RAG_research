"""
app.py — Streamlit UI for the Research Paper Q&A Assistant.

Layout
──────
┌─ Sidebar ──────────────────────────────────────────────┐
│  • Upload documents  (PDF / TXT / MD)                  │
│  • Index status  (doc count, chunk count)              │
│  • Retrieval settings  (top-k, threshold slider)       │
│  • Reset index button                                  │
└────────────────────────────────────────────────────────┘
┌─ Main panel ───────────────────────────────────────────┐
│  Chat history  (user + assistant bubbles)              │
│  Query input box                                       │
│  ── per-answer expanders ──                            │
│    📚 Bibliography                                     │
│    🔍 Retrieved Chunks                                 │
│    ⚙  Metadata  (tokens, latency, grounding flags)    │
└────────────────────────────────────────────────────────┘

Run
───
streamlit run app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

# Ensure project root is importable when running via `streamlit run app.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_app.pipeline import RAGPipeline, RAGResponse
from rag_app.ingestion import ingest_file
from rag_app.vector_store import VectorStore
from rag_app.config import TOP_K_RESULTS, SIMILARITY_THRESHOLD


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Research Q&A Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session state initialisation ──────────────────────────────────────────────

def _init_state() -> None:
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RAGPipeline()
    if "chat_history" not in st.session_state:
        # list of {"role": "user"|"assistant", "content": str, "response": RAGResponse|None}
        st.session_state.chat_history = []


_init_state()
pipeline: RAGPipeline = st.session_state.pipeline


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔬 Research Q&A")
    st.caption("RAG-powered assistant grounded in your documents.")
    st.divider()

    # ── Upload documents ──────────────────────────────────────────────────────
    st.subheader("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        label="Drop PDF, TXT, or MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Files are chunked, embedded, and stored in ChromaDB.",
    )

    if uploaded_files:
        if st.button("📥 Ingest uploaded files", use_container_width=True):
            store: VectorStore = pipeline._store
            progress = st.progress(0, text="Ingesting…")
            total_chunks = 0
            for i, uf in enumerate(uploaded_files):
                suffix = Path(uf.name).suffix
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as tmp:
                    tmp.write(uf.read())
                    tmp_path = tmp.name
                try:
                    n = ingest_file(tmp_path, store=store)
                    total_chunks += n
                    st.toast(f"✅ {uf.name} → {n} chunks", icon="📄")
                except Exception as exc:
                    st.toast(f"❌ {uf.name}: {exc}", icon="⚠️")
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
                progress.progress((i + 1) / len(uploaded_files))
            progress.empty()
            st.success(f"Ingested {total_chunks} total chunks from {len(uploaded_files)} file(s).")
            st.rerun()

    st.divider()

    # ── Index status ──────────────────────────────────────────────────────────
    st.subheader("🗂 Index Status")
    status = pipeline.index_status()
    col1, col2 = st.columns(2)
    col1.metric("Documents", status["total_documents"])
    col2.metric("Chunks", status["total_chunks"])

    if status["documents"]:
        with st.expander("Indexed files", expanded=False):
            for d in status["documents"]:
                st.markdown(f"- `{Path(d).name}`")
    else:
        st.info("No documents indexed yet. Upload files above.")

    st.divider()

    # ── Retrieval settings ────────────────────────────────────────────────────
    st.subheader("⚙️ Retrieval Settings")
    top_k = st.slider(
        "Top-k chunks", min_value=1, max_value=15,
        value=TOP_K_RESULTS, step=1,
    )
    threshold = st.slider(
        "Min similarity", min_value=0.0, max_value=1.0,
        value=float(SIMILARITY_THRESHOLD), step=0.05,
    )

    st.divider()

    # ── Danger zone ───────────────────────────────────────────────────────────
    st.subheader("🗑 Danger Zone")
    if st.button("Reset index (delete all chunks)", use_container_width=True, type="secondary"):
        pipeline._store.reset()
        # Reinitialise so the pipeline picks up the fresh collection
        st.session_state.pipeline = RAGPipeline()
        pipeline = st.session_state.pipeline
        st.warning("Index has been reset.")
        st.rerun()


# ── Main panel ────────────────────────────────────────────────────────────────

st.title("🔬 Research Paper Q&A Assistant")
st.caption(
    "Ask questions grounded strictly in your uploaded research documents. "
    "Every answer is citation-backed and hallucination-checked."
)

# ── Chat history display ──────────────────────────────────────────────────────

for entry in st.session_state.chat_history:
    role    = entry["role"]
    content = entry["content"]
    resp: RAGResponse | None = entry.get("response")

    with st.chat_message(role):
        st.markdown(content)

        if role == "assistant" and resp is not None:
            # Quality badges
            badge_cols = st.columns(3)
            badge_cols[0].badge(
                "✅ Grounded" if resp.is_grounded else "⚠️ Possibly ungrounded",
                color="green" if resp.is_grounded else "orange",
            )
            badge_cols[1].badge(
                "🚫 Refused" if resp.is_refused else "💬 Answered",
                color="red" if resp.is_refused else "blue",
            )
            badge_cols[2].badge(
                f"⚡ {resp.latency_seconds:.2f}s",
                color="gray",
            )

            # Warnings inline
            for w in resp.warnings:
                st.warning(w, icon="⚠️")

            # Expanders
            if resp.bibliography:
                with st.expander("📚 Bibliography", expanded=False):
                    st.code(resp.bibliography, language=None)

            if resp.results:
                with st.expander(f"🔍 Retrieved Chunks ({len(resp.results)})", expanded=False):
                    for i, r in enumerate(resp.results, 1):
                        st.markdown(
                            f"**[Doc {i}]** `{Path(r.source).name}` "
                            f"p.{r.page} | score `{r.score:.2f}`"
                        )
                        st.caption(r.text[:300] + ("…" if len(r.text) > 300 else ""))
                        st.divider()

            with st.expander("⚙️ Metadata", expanded=False):
                meta_cols = st.columns(4)
                meta_cols[0].metric("Prompt tokens",     resp.prompt_tokens)
                meta_cols[1].metric("Completion tokens", resp.completion_tokens)
                meta_cols[2].metric("Total tokens",      resp.total_tokens)
                meta_cols[3].metric("Chunks retrieved",  len(resp.results))


# ── Query input ───────────────────────────────────────────────────────────────

query = st.chat_input(
    placeholder="Ask a question about your research documents…",
    disabled=(pipeline.index_status()["total_chunks"] == 0),
)

if pipeline.index_status()["total_chunks"] == 0:
    st.info("Upload and ingest at least one document before asking questions.", icon="💡")

if query:
    # Display user message
    st.session_state.chat_history.append(
        {"role": "user", "content": query, "response": None}
    )

    with st.chat_message("user"):
        st.markdown(query)

    # Run pipeline
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating answer…"):
            try:
                resp: RAGResponse = pipeline.ask(
                    query=query,
                    top_k=top_k,
                    threshold=threshold,
                )
                answer_text = resp.annotated_answer
            except Exception as exc:
                answer_text = f"❌ Pipeline error: {exc}"
                resp = None  # type: ignore[assignment]

        st.markdown(answer_text)

        if resp is not None:
            badge_cols = st.columns(3)
            badge_cols[0].badge(
                "✅ Grounded" if resp.is_grounded else "⚠️ Possibly ungrounded",
                color="green" if resp.is_grounded else "orange",
            )
            badge_cols[1].badge(
                "🚫 Refused" if resp.is_refused else "💬 Answered",
                color="red" if resp.is_refused else "blue",
            )
            badge_cols[2].badge(f"⚡ {resp.latency_seconds:.2f}s", color="gray")

            for w in resp.warnings:
                st.warning(w, icon="⚠️")

            if resp.bibliography:
                with st.expander("📚 Bibliography", expanded=True):
                    st.code(resp.bibliography, language=None)

            if resp.results:
                with st.expander(f"🔍 Retrieved Chunks ({len(resp.results)})", expanded=False):
                    for i, r in enumerate(resp.results, 1):
                        st.markdown(
                            f"**[Doc {i}]** `{Path(r.source).name}` "
                            f"p.{r.page} | score `{r.score:.2f}`"
                        )
                        st.caption(r.text[:300] + ("…" if len(r.text) > 300 else ""))
                        st.divider()

            with st.expander("⚙️ Metadata", expanded=False):
                meta_cols = st.columns(4)
                meta_cols[0].metric("Prompt tokens",     resp.prompt_tokens)
                meta_cols[1].metric("Completion tokens", resp.completion_tokens)
                meta_cols[2].metric("Total tokens",      resp.total_tokens)
                meta_cols[3].metric("Chunks retrieved",  len(resp.results))

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer_text, "response": resp}
    )
