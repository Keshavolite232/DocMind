"""
Tests for the RAG engine.
Run: pytest tests/ -v
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_anthropic_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-fake-key")


@pytest.fixture
def sample_pdf_path(tmp_path):
    """Create a minimal valid PDF for testing."""
    pdf_content = b"""%PDF-1.4
1 0 obj<</Type /Catalog /Pages 2 0 R>>endobj
2 0 obj<</Type /Pages /Kids [3 0 R] /Count 1>>endobj
3 0 obj<</Type /Page /MediaBox [0 0 612 792] /Parent 2 0 R
  /Contents 4 0 R /Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 44>>stream
BT /F1 12 Tf 100 700 Td (Hello RAG World) Tj ET
endstream endobj
5 0 obj<</Type /Font /Subtype /Type1 /BaseFont /Helvetica>>endobj
xref 0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000360 00000 n
trailer<</Size 6 /Root 1 0 R>>
startxref 441
%%EOF"""
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(pdf_content)
    return str(pdf_file)


# ── Engine initialisation ──────────────────────────────────────────────────────

class TestRAGEngineInit:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with (
            patch("app.rag_engine.FastEmbedEmbeddings"),
            patch("app.rag_engine.ChatAnthropic"),
            patch("app.rag_engine.Chroma"),
        ):
            from app.rag_engine import RAGEngine
            with pytest.raises(ValueError, match="Anthropic API key required"):
                RAGEngine()

    def test_invalid_vector_store(self, mock_anthropic_key):
        with (
            patch("app.rag_engine.FastEmbedEmbeddings"),
            patch("app.rag_engine.ChatAnthropic"),
            patch("app.rag_engine.Chroma"),
        ):
            from app.rag_engine import RAGEngine
            with pytest.raises(ValueError, match="Unknown vector store"):
                RAGEngine(vector_store="faiss")


# ── Ingestion ──────────────────────────────────────────────────────────────────

class TestIngestion:
    @patch("app.rag_engine.create_retrieval_chain")
    @patch("app.rag_engine.create_history_aware_retriever")
    @patch("app.rag_engine.Chroma")
    @patch("app.rag_engine.ChatAnthropic")
    @patch("app.rag_engine.FastEmbedEmbeddings")
    @patch("app.rag_engine.PyPDFLoader")
    def test_ingest_pdf_success(
        self, mock_loader, mock_embeddings, mock_llm, mock_chroma,
        mock_hist_retriever, mock_retrieval_chain, mock_anthropic_key, sample_pdf_path
    ):
        mock_loader.return_value.load.return_value = [
            Document(page_content="Hello RAG World", metadata={"page": 0})
        ]
        mock_chroma_instance = MagicMock()
        mock_chroma.return_value = mock_chroma_instance

        from app.rag_engine import RAGEngine
        engine = RAGEngine()
        result = engine.ingest_pdf(sample_pdf_path)

        assert result["status"] == "success"
        assert result["pages"] == 1
        assert result["chunks"] >= 1
        mock_chroma_instance.add_documents.assert_called_once()

    @patch("app.rag_engine.create_retrieval_chain")
    @patch("app.rag_engine.create_history_aware_retriever")
    @patch("app.rag_engine.Chroma")
    @patch("app.rag_engine.ChatAnthropic")
    @patch("app.rag_engine.FastEmbedEmbeddings")
    @patch("app.rag_engine.PyPDFLoader")
    def test_duplicate_ingest_skipped(
        self, mock_loader, mock_embeddings, mock_llm, mock_chroma,
        mock_hist_retriever, mock_retrieval_chain, mock_anthropic_key, sample_pdf_path
    ):
        mock_loader.return_value.load.return_value = [
            Document(page_content="Hello RAG World", metadata={"page": 0})
        ]
        from app.rag_engine import RAGEngine
        engine = RAGEngine()
        engine.ingest_pdf(sample_pdf_path)
        result = engine.ingest_pdf(sample_pdf_path)   # second call

        assert result["status"] == "skipped"
        assert result["reason"] == "already ingested"

    def test_ingest_missing_file(self, mock_anthropic_key):
        with (
            patch("app.rag_engine.FastEmbedEmbeddings"),
            patch("app.rag_engine.ChatAnthropic"),
            patch("app.rag_engine.Chroma"),
        ):
            from app.rag_engine import RAGEngine
            engine = RAGEngine()
            with pytest.raises(FileNotFoundError):
                engine.ingest_pdf("/nonexistent/file.pdf")


# ── Querying ───────────────────────────────────────────────────────────────────

class TestQuerying:
    @patch("app.rag_engine.Chroma")
    @patch("app.rag_engine.ChatAnthropic")
    @patch("app.rag_engine.FastEmbedEmbeddings")
    def test_query_without_docs_raises(self, mock_embeddings, mock_llm, mock_chroma, mock_anthropic_key):
        from app.rag_engine import RAGEngine
        engine = RAGEngine()
        with pytest.raises(RuntimeError, match="No documents ingested"):
            engine.query("What is this about?")

    @patch("app.rag_engine.Chroma")
    @patch("app.rag_engine.ChatAnthropic")
    @patch("app.rag_engine.FastEmbedEmbeddings")
    def test_query_returns_answer_and_sources(
        self, mock_embeddings, mock_llm, mock_chroma, mock_anthropic_key
    ):
        from app.rag_engine import RAGEngine

        # Chain now uses .invoke() and returns "context" (not "source_documents")
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "answer": "The answer is 42.",
            "context": [
                Document(
                    page_content="The answer to life is 42.",
                    metadata={"source_file": "deep_thought.pdf", "page": 7}
                )
            ],
        }

        engine = RAGEngine()
        engine.chain = mock_chain

        result = engine.query("What is the answer?")
        assert result["answer"] == "The answer is 42."
        assert len(result["sources"]) == 1
        assert result["sources"][0]["file"] == "deep_thought.pdf"


# ── Memory ─────────────────────────────────────────────────────────────────────

class TestMemory:
    @patch("app.rag_engine.Chroma")
    @patch("app.rag_engine.ChatAnthropic")
    @patch("app.rag_engine.FastEmbedEmbeddings")
    def test_clear_memory(self, mock_embeddings, mock_llm, mock_chroma, mock_anthropic_key):
        from app.rag_engine import RAGEngine

        engine = RAGEngine()
        # Add messages directly to the InMemoryChatMessageHistory
        engine.chat_history.add_user_message("Hi")
        engine.chat_history.add_ai_message("Hello!")
        assert len(engine.chat_history.messages) == 2

        engine.clear_memory()
        assert engine.chat_history.messages == []
