"""
RAG Engine — Core document ingestion, embedding, retrieval, and generation logic.
Supports both ChromaDB (local) and Pinecone (cloud) as vector stores.
"""

import os
import hashlib
import logging
import time
from pathlib import Path
from typing import Optional, Callable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_core.embeddings import Embeddings


class ChromaDefaultEmbeddings(Embeddings):
    """
    Thin LangChain wrapper around chromadb's built-in ONNX embedding function
    (all-MiniLM-L6-v2 in ONNX format).  No PyTorch, no Rust — just onnxruntime
    which chromadb already depends on.
    """
    def __init__(self):
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(v) for v in e] for e in self._fn(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [float(v) for v in self._fn([text])[0]]

from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser


# ── Prompt templates ───────────────────────────────────────────────────────────

CONTEXTUALIZE_Q_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Given the chat history and the latest user question, "
        "formulate a standalone question that can be understood without the chat history. "
        "Do NOT answer — just reformulate if needed, otherwise return it as is."
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a precise, helpful assistant that answers questions based strictly "
        "on the provided document context. If the context doesn't contain enough "
        "information to answer, say so clearly rather than hallucinating.\n\n"
        "Context from documents:\n{context}"
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


# ── RAG Engine ─────────────────────────────────────────────────────────────────

class RAGEngine:
    """
    Orchestrates the full RAG pipeline:
      1. Load & split PDF documents
      2. Embed chunks into a vector store
      3. Answer questions with conversational memory (LCEL-based)
    """

    def __init__(
        self,
        vector_store: str = "chroma",
        chroma_persist_dir: str = "./chroma_db",
        pinecone_index: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        embedding_model: str = "default",
        model_name: str = "claude-sonnet-4-5",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        retriever_k: int = 6,
        memory_window: int = 10,
    ):
        self.vector_store_type = vector_store
        self.chroma_persist_dir = chroma_persist_dir
        self.pinecone_index = pinecone_index
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retriever_k = retriever_k
        self.memory_window = memory_window

        anthropic_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_key:
            raise ValueError("Anthropic API key required — set ANTHROPIC_API_KEY env var.")

        log.info("INIT ▶ Loading chromadb ONNX embedding model")
        t = time.perf_counter()
        self.embeddings = ChromaDefaultEmbeddings()
        log.info("INIT ✓ Embeddings ready (%.1fs)", time.perf_counter() - t)

        log.info("INIT ▶ Initialising LLM (%s)", model_name)
        t = time.perf_counter()
        self.llm = ChatAnthropic(
            model=model_name,
            temperature=0,
            anthropic_api_key=anthropic_key,
        )
        log.info("INIT ✓ LLM ready (%.1fs)", time.perf_counter() - t)

        log.info("INIT ▶ Initialising vector store (%s)", vector_store)
        t = time.perf_counter()
        self.vector_store = self._init_vector_store()
        log.info("INIT ✓ Vector store ready (%.1fs)", time.perf_counter() - t)

        self.chat_history = InMemoryChatMessageHistory()
        self.chain: Optional[RunnableWithMessageHistory] = None
        self.ingested_files: set[str] = set()
        log.info("INIT ✓ Engine fully initialised")

    # ── Vector store initialisation ────────────────────────────────────────────

    def _init_vector_store(self):
        if self.vector_store_type == "chroma":
            return InMemoryVectorStore(embedding=self.embeddings)
        elif self.vector_store_type == "pinecone":
            return self._init_pinecone()
        else:
            raise ValueError(f"Unknown vector store: {self.vector_store_type}")

    def _init_pinecone(self):
        try:
            from langchain_pinecone import PineconeVectorStore
            import pinecone
            pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index_name = self.pinecone_index or os.getenv("PINECONE_INDEX", "rag-index")
            return PineconeVectorStore(
                index=pc.Index(index_name),
                embedding=self.embeddings,
            )
        except ImportError:
            raise ImportError(
                "Install langchain-pinecone and pinecone-client for Pinecone support."
            )

    # ── Document ingestion ─────────────────────────────────────────────────────

    def ingest_pdf(self, pdf_path: str, progress_callback: Optional[Callable] = None,
                   display_name: Optional[str] = None) -> dict:
        """
        Load a PDF, split into chunks, embed, and add to the vector store.
        display_name overrides path.name in chunk metadata (use when pdf_path is a tmp file).
        progress_callback(step, total_steps, message) is called at each stage.
        """
        path = Path(pdf_path)
        total_steps = 4

        def report(step: int, message: str):
            log.info(message)
            if progress_callback:
                progress_callback(step, total_steps, message)

        log.info("INGEST ▶ Starting: %s", path.name)

        if not path.exists():
            log.error("INGEST ✗ File not found: %s", pdf_path)
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        file_hash = self._file_hash(path)
        if file_hash in self.ingested_files:
            log.info("INGEST ↩ Skipped (already ingested): %s", path.name)
            return {"status": "skipped", "reason": "already ingested", "file": path.name}

        report(1, "📄 Loading PDF pages...")
        t = time.perf_counter()
        loader = PyPDFLoader(str(path))
        raw_docs = loader.load()
        log.info("INGEST ✓ Step 1/4 — Loaded %d pages (%.1fs)", len(raw_docs), time.perf_counter() - t)

        report(2, f"✂️ Splitting into chunks... ({len(raw_docs)} pages)")
        t = time.perf_counter()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(raw_docs)
        label = display_name or path.name
        for chunk in chunks:
            chunk.metadata["source_file"] = label
            chunk.metadata["file_hash"] = file_hash
        log.info("INGEST ✓ Step 2/4 — Created %d chunks (%.1fs)", len(chunks), time.perf_counter() - t)

        report(3, f"🔢 Generating embeddings... ({len(chunks)} chunks)")
        t = time.perf_counter()
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            self.vector_store.add_documents(chunks[i : i + batch_size])
        log.info("INGEST ✓ Step 3/4 — Stored embeddings (%.1fs)", time.perf_counter() - t)

        report(4, "🔗 Building retrieval chain...")
        t = time.perf_counter()
        self.ingested_files.add(file_hash)
        self._refresh_chain()
        log.info("INGEST ✓ Step 4/4 — Chain ready (%.1fs)", time.perf_counter() - t)

        log.info("INGEST ✓ Complete: %s — %d pages, %d chunks", path.name, len(raw_docs), len(chunks))
        return {
            "status": "success",
            "file": path.name,
            "pages": len(raw_docs),
            "chunks": len(chunks),
        }

    def ingest_directory(self, dir_path: str) -> list[dict]:
        """Ingest all PDFs in a directory."""
        results = []
        for pdf in Path(dir_path).glob("*.pdf"):
            results.append(self.ingest_pdf(str(pdf)))
        return results

    # ── Querying ───────────────────────────────────────────────────────────────

    def query(self, question: str) -> dict:
        """Answer a question using the RAG pipeline with conversational memory.
        Retries up to 3 times on rate limit / overload errors."""
        if not self.chain:
            raise RuntimeError("No documents ingested yet. Please upload PDFs first.")

        # Trim to memory window (keep last k turns = k*2 messages)
        msgs = self.chat_history.messages
        if len(msgs) > self.memory_window * 2:
            self.chat_history.messages = msgs[-(self.memory_window * 2):]

        last_error = None
        for attempt in range(3):
            try:
                result = self.chain.invoke(
                    {"input": question},
                    config={"configurable": {"session_id": "default"}},
                )
                break
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                retryable = any(x in err_str for x in [
                    "rate_limit", "rate limit", "429",
                    "overloaded", "too many requests", "capacity",
                ])
                if retryable and attempt < 2:
                    wait = 2 ** attempt  # 1s, 2s
                    log.warning(
                        "Rate limit / overload on attempt %d/3 — retrying in %ds",
                        attempt + 1, wait,
                    )
                    time.sleep(wait)
                    continue
                raise
        else:
            raise last_error

        sources = []
        for doc in result.get("context", []):
            sources.append({
                "file": doc.metadata.get("source_file", "unknown"),
                "page": doc.metadata.get("page", "?"),
                "snippet": doc.page_content[:200].strip(),
            })

        return {
            "answer": result["answer"],
            "sources": sources,
        }

    def clear_memory(self):
        """Reset conversation history."""
        self.chat_history.clear()

    # ── Internals ──────────────────────────────────────────────────────────────

    def _refresh_chain(self):
        """Rebuild the LCEL retrieval chain after new docs are ingested."""
        log.info("CHAIN ▶ Building history-aware retriever")
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.retriever_k},
        )

        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, CONTEXTUALIZE_Q_PROMPT
        )

        qa_chain = create_stuff_documents_chain(self.llm, QA_PROMPT)
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        self.chain = RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: self.chat_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        log.info("CHAIN ✓ Chain built successfully")

    @staticmethod
    def _file_hash(path: Path) -> str:
        return hashlib.md5(path.read_bytes()).hexdigest()
