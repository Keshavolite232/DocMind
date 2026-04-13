"""
RAG Engine — Core document ingestion, embedding, retrieval, and generation logic.
Supports both ChromaDB (local) and Pinecone (cloud) as vector stores.
"""

import os
import hashlib
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser


# ── Prompt templates ───────────────────────────────────────────────────────────

# Rephrases the user's question into a standalone question given chat history,
# so the retriever doesn't need to understand conversational context.
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
      2. Embed chunks into a vector store (ChromaDB or Pinecone)
      3. Answer questions with conversational memory (LCEL-based)
    """

    def __init__(
        self,
        vector_store: str = "chroma",
        chroma_persist_dir: str = "./chroma_db",
        pinecone_index: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_name: str = "claude-sonnet-4-5",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        retriever_k: int = 4,
        memory_window: int = 5,
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

        # Embeddings run locally via sentence-transformers — no API key needed
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.llm = ChatAnthropic(
            model=model_name,
            temperature=0,
            anthropic_api_key=anthropic_key,
        )

        self.vector_store = self._init_vector_store()
        self.chat_history = InMemoryChatMessageHistory()
        self.chain: Optional[RunnableWithMessageHistory] = None
        self.ingested_files: set[str] = set()

    # ── Vector store initialisation ────────────────────────────────────────────

    def _init_vector_store(self):
        if self.vector_store_type == "chroma":
            return Chroma(
                persist_directory=self.chroma_persist_dir,
                embedding_function=self.embeddings,
                collection_name="rag_docs",
            )
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

    def ingest_pdf(self, pdf_path: str) -> dict:
        """Load a PDF, split into chunks, embed, and add to the vector store."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        file_hash = self._file_hash(path)
        if file_hash in self.ingested_files:
            return {"status": "skipped", "reason": "already ingested", "file": path.name}

        loader = PyPDFLoader(str(path))
        raw_docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(raw_docs)

        for chunk in chunks:
            chunk.metadata["source_file"] = path.name
            chunk.metadata["file_hash"] = file_hash

        # chromadb>=0.5.0 auto-persists — .persist() was removed
        self.vector_store.add_documents(chunks)
        self.ingested_files.add(file_hash)
        self._refresh_chain()

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
        """Answer a question using the RAG pipeline with conversational memory."""
        if not self.chain:
            raise RuntimeError("No documents ingested yet. Please upload PDFs first.")

        # Trim to memory window (keep last k turns = k*2 messages)
        msgs = self.chat_history.messages
        if len(msgs) > self.memory_window * 2:
            self.chat_history.messages = msgs[-(self.memory_window * 2):]

        result = self.chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": "default"}},
        )

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
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": self.retriever_k, "fetch_k": self.retriever_k * 3},
        )

        # Retriever that rephrases the question using chat history before searching
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, CONTEXTUALIZE_Q_PROMPT
        )

        # Combine retrieved docs into an answer
        qa_chain = create_stuff_documents_chain(self.llm, QA_PROMPT)

        # Full RAG chain: history-aware retrieval → answer generation
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        # Wrap with persistent message history
        self.chain = RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: self.chat_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    @staticmethod
    def _file_hash(path: Path) -> str:
        return hashlib.md5(path.read_bytes()).hexdigest()
