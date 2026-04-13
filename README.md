# 📄 DocMind — RAG-Powered PDF Q&A Chatbot

A production-quality Retrieval-Augmented Generation (RAG) system that lets you chat with your PDF documents using natural language. Built with **LangChain**, **ChromaDB** (or **Pinecone**), and **Claude (Anthropic)** as the LLM, with a clean Streamlit UI.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green)
![Claude](https://img.shields.io/badge/LLM-Claude%20(Anthropic)-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

- **Multi-document ingestion** — upload and index multiple PDFs at once
- **Conversational memory** — follow-up questions understand prior context
- **Source attribution** — every answer cites the page and file it came from
- **Dual vector store** — swap between ChromaDB (local) and Pinecone (cloud) with one config flag
- **Duplicate detection** — re-ingesting the same PDF is safely skipped
- **MMR retrieval** — Maximal Marginal Relevance prevents repetitive context chunks
- **CLI interface** — batch ingest and query without the UI
- **Fully tested** — unit tests with mocked external calls

---

## 🏗 Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────────────┐
│               RAG Pipeline                   │
│                                              │
│  PDF ──► Splitter ──► Embeddings             │
│                           │                  │
│                     Vector Store             │
│                    (Chroma / Pinecone)        │
│                           │                  │
│  Question ──► Retriever ──► Top-k Chunks     │
│                                  │           │
│              Prompt + Context ──► LLM        │
│                                  │           │
│                  Answer + Sources ◄──────────┘
└─────────────────────────────────────────────┘
```

**Key design choices:**
- `RecursiveCharacterTextSplitter` with 1000-token chunks and 200-token overlap preserves sentence coherence
- MMR retrieval (`search_type="mmr"`) reduces redundant context vs. plain cosine similarity
- `ConversationBufferWindowMemory(k=5)` keeps recent turns without ballooning the prompt
- Engine is initialized once and cached via `@st.cache_resource` to avoid re-embedding on every interaction

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/rag-pdf-chatbot.git
cd rag-pdf-chatbot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Launch the Streamlit UI

```bash
streamlit run app/ui.py
```

Open [http://localhost:8501](http://localhost:8501), paste your API key in the sidebar, upload PDFs, and start chatting.

---

## 🖥 CLI Usage

```bash
# Ingest a single PDF
python cli.py ingest --path ./docs/report.pdf

# Ingest a whole folder
python cli.py ingest --dir ./docs/

# One-shot question
python cli.py query "What are the key findings?"

# Interactive chat
python cli.py chat
```

---

## 🔧 Configuration

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required. Your Anthropic API key. |
| `OPENAI_API_KEY` | — | Required. Used for text embeddings. |
| `VECTOR_STORE` | `chroma` | `chroma` (local) or `pinecone` (cloud) |
| `PINECONE_API_KEY` | — | Required if using Pinecone |
| `PINECONE_INDEX` | `rag-index` | Pinecone index name |

### Switching to Pinecone

1. Uncomment the Pinecone packages in `requirements.txt` and `pip install`
2. Set `PINECONE_API_KEY` and `PINECONE_INDEX` in `.env`
3. Select "pinecone" in the UI dropdown (or set `VECTOR_STORE=pinecone`)

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Tests mock all external API calls — no OpenAI key or vector DB needed.

---

## 📁 Project Structure

```
rag-pdf-chatbot/
├── app/
│   ├── __init__.py
│   ├── rag_engine.py     # Core RAG logic (ingest, embed, retrieve, generate)
│   └── ui.py             # Streamlit frontend
├── tests/
│   └── test_rag_engine.py
├── cli.py                # Command-line interface
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🔮 Possible Extensions

- [ ] Re-ranking with a cross-encoder (Cohere Rerank / `flashrank`)
- [ ] Hybrid search (BM25 sparse + dense vector)
- [ ] Document comparison mode — diff two PDFs
- [ ] Export conversation as PDF report
- [ ] Async streaming responses
- [ ] LangSmith tracing for observability

---

## 📜 License

MIT — free to use, modify, and distribute.
