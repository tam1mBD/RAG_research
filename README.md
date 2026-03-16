# Research Paper Q&A Assistant (RAG)

A production-ready **Retrieval-Augmented Generation** system that answers questions grounded strictly in your own uploaded research papers — with citation-backed answers, hallucination detection, and a Streamlit chat UI.

---

## Architecture

```
User Query
   │
   ▼
[1] Embedder           cached SHA-256 disk cache → local sentence-transformers
   │
   ▼
[2] Vector Store       ChromaDB persistent collection (cosine similarity)
   │
   ▼
[3] Prompt Builder     5-layer structured prompt
   │                     Layer 1 — System role definition
   │                     Layer 2 — Grounding & refusal rules
   │                     Layer 3 — Few-shot examples
   │                     Layer 4 — Retrieved document context
   │                     Layer 5 — User query + format constraints
   ▼
[4] LLM Client         LangChain ChatGroq/ChatOpenAI (provider switch) + grounding check
   │
   ▼
[5] Citation Mapper    [Doc N, p.X] → source file, page, section, snippet
   │
   ▼
RAGResponse            answer · annotated_answer · bibliography · warnings · tokens
```

---

## Project Structure

```
RAG_research/
├── app.py                      # Streamlit UI entry point
├── requirements.txt
├── pytest.ini
├── .env.example                # Copy to .env and fill in your API key
│
├── rag_app/
│   ├── config.py               # Centralised settings from env vars
│   ├── logger.py               # Shared logging
│   ├── embedder.py             # Embedding with SHA-256 disk cache
│   ├── vector_store.py         # ChromaDB wrapper (add / query / delete)
│   ├── ingestion.py            # LangChain loaders + RecursiveCharacterTextSplitter
│   ├── prompt_builder.py       # Multi-layer prompt assembly + token budget
│   ├── llm_client.py           # LangChain LLM + grounding validator
│   ├── citation_mapper.py      # [Doc N] → source resolution + bibliography
│   └── pipeline.py             # Orchestrates all 5 steps → RAGResponse
│
├── scripts/
│   ├── ingest.py               # CLI: ingest PDF/TXT/MD into ChromaDB
│   ├── ask.py                  # CLI: ask a question without the UI
│   └── benchmark.py            # Accuracy & hallucination benchmark runner
│
├── tests/
│   ├── conftest.py
│   ├── test_embedder.py
│   ├── test_prompt_builder.py
│   ├── test_citation_mapper.py
│   └── test_llm_client.py
│
└── data/
    ├── docs/                   # Drop your PDF / TXT / MD files here
    ├── chroma_db/              # ChromaDB persistence (auto-created)
    └── embedding_cache/        # Embedding JSON cache (auto-created)
```

---

## Quick Start

### 1. Clone & create virtual environment

```bash
git clone https://github.com/tam1mBD/RAG_research.git
cd RAG_research
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and set:
#   LLM_PROVIDER=groq  + GROQ_API_KEY
#   OR
#   LLM_PROVIDER=openai + OPENAI_API_KEY
```

### 4. Ingest documents

Drop your PDF / TXT / MD research papers into `data/docs/`, then:

```bash
python scripts/ingest.py                   # ingest all files in data/docs/
python scripts/ingest.py --file paper.pdf  # single file
python scripts/ingest.py --reset           # wipe index and re-ingest
```

### 5. Ask questions

```bash
# CLI
python scripts/ask.py "What is the attention mechanism?"
python scripts/ask.py "Summarise the results" --top-k 8 --threshold 0.4

# UI
streamlit run app.py
```

### 6. Run tests

```bash
pytest
```

### 7. Run benchmark

```bash
python scripts/benchmark.py
python scripts/benchmark.py --output results/report.json
```

---

## Key Features

| Feature | Detail |
|---|---|
| **5-layer prompt** | System role → task rules → few-shot → context → query |
| **Grounding rules** | Strict context-only sourcing, explicit refusal phrase |
| **Refusal logic** | Out-of-context queries get a standardised refusal response |
| **Citation mapping** | Every `[Doc N, p.X]` resolved to filename, page, section, snippet |
| **Hallucination detection** | Post-generation regex check; flags uncited long answers |
| **Embedding cache** | SHA-256 disk cache per text + model — zero duplicate API calls |
| **Token budget** | Tiktoken-based trimming keeps every prompt within `MAX_PROMPT_TOKENS` |
| **Chunk upsert** | Safe re-ingestion — existing chunks are overwritten not duplicated |
| **Benchmark suite** | 8 built-in questions across 5 categories with ASCII report |

---

## Configuration (`rag_app/config.py`)

All values can be overridden via `.env`:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `groq` or `openai` |
| `GROQ_API_KEY` | — | Required when `LLM_PROVIDER=groq` |
| `OPENAI_API_KEY` | — | Required when `LLM_PROVIDER=openai` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformers embedding model |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Chat model name for selected provider |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | Chunks retrieved per query |
| `SIMILARITY_THRESHOLD` | `0.30` | Min cosine score to include chunk |
| `MAX_PROMPT_TOKENS` | `3500` | Total prompt token budget |
| `MAX_RESPONSE_TOKENS` | `600` | Max LLM response length |

---

## Prompt Quality Improvements

1. **Strict grounding** — model is forbidden from using any knowledge outside the context block
2. **Explicit refusal rule** — standardised phrase for out-of-context questions, verified post-generation
3. **Few-shot examples** — 3 demonstrations: grounded answer, refusal, conflicting sources
4. **Format constraints** — ≤5 sentences, cite every claim with `[Doc N, p.X]`
5. **Token optimisation** — tiktoken budget enforced; lowest-relevance chunks dropped first
