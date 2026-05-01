import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from query import RAGAssistant


class MockEmbeddingService:
    def embed(self, text):
        return [0.1, 0.2, 0.3]


class MockVectorStore:
    def search(self, query_embedding, n_results=3):
        return {
            "documents": [[
                "Employees receive paid holidays including New Year's Day and Christmas Day."
            ]],
            "metadatas": [[
                {
                    "source": "employee_handbook.txt",
                    "chunk_id": 1
                }
            ]]
        }


class MockLLMService:
    def generate(self, query_text, context):
        return "Employees receive paid holidays."


class MockFormatter:
    def format(self, results):
        return [
            {
                "source": "employee_handbook.txt",
                "chunk_id": 1,
                "text": "Employees receive paid holidays including New Year's Day and Christmas Day."
            }
        ]


class MockContextBuilder:
    def build(self, chunks):
        return chunks[0]["text"]


def test_rag_assistant_returns_answer():
    assistant = RAGAssistant(
        embedding_service=MockEmbeddingService(),
        vector_store=MockVectorStore(),
        llm_service=MockLLMService(),
        formatter=MockFormatter(),
        context_builder=MockContextBuilder()
    )

    result = assistant.ask_question("What holidays do employees get?")

    assert result["answer"] == "Employees receive paid holidays."
    assert len(result["chunks"]) == 1
    assert result["chunks"][0]["source"] == "employee_handbook.txt"


def test_rag_assistant_handles_no_chunks():
    class EmptyFormatter:
        def format(self, results):
            return []

    assistant = RAGAssistant(
        embedding_service=MockEmbeddingService(),
        vector_store=MockVectorStore(),
        llm_service=MockLLMService(),
        formatter=EmptyFormatter(),
        context_builder=MockContextBuilder()
    )

    result = assistant.ask_question("Unknown question")

    assert result["answer"] == "I do not know based on the provided documents."
    assert result["chunks"] == []


def test_rag_assistant_passes_n_results_to_vector_store():
    class TrackingVectorStore:
        def __init__(self):
            self.received_n_results = None

        def search(self, query_embedding, n_results=3):
            self.received_n_results = n_results
            return {
                "documents": [["sample text"]],
                "metadatas": [[{"source": "doc.txt", "chunk_id": 0}]]
            }

    vector_store = TrackingVectorStore()

    assistant = RAGAssistant(
        embedding_service=MockEmbeddingService(),
        vector_store=vector_store,
        llm_service=MockLLMService(),
        formatter=MockFormatter(),
        context_builder=MockContextBuilder()
    )

    assistant.ask_question("test question", n_results=5)

    assert vector_store.received_n_results == 5


def test_rag_assistant_passes_query_to_embedding_service():
    class TrackingEmbeddingService:
        def __init__(self):
            self.received_text = None

        def embed(self, text):
            self.received_text = text
            return [0.1, 0.2, 0.3]

    embedding_service = TrackingEmbeddingService()

    assistant = RAGAssistant(
        embedding_service=embedding_service,
        vector_store=MockVectorStore(),
        llm_service=MockLLMService(),
        formatter=MockFormatter(),
        context_builder=MockContextBuilder()
    )

    assistant.ask_question("What is PTO?")

    assert embedding_service.received_text == "What is PTO?"


def test_rag_assistant_passes_context_to_llm_service():
    class TrackingLLMService:
        def __init__(self):
            self.received_context = None

        def generate(self, query_text, context):
            self.received_context = context
            return "tracked answer"

    llm_service = TrackingLLMService()

    assistant = RAGAssistant(
        embedding_service=MockEmbeddingService(),
        vector_store=MockVectorStore(),
        llm_service=llm_service,
        formatter=MockFormatter(),
        context_builder=MockContextBuilder()
    )

    assistant.ask_question("What is PTO?")

    assert llm_service.received_context == (
        "Employees receive paid holidays including New Year's Day and Christmas Day."
    )


def test_rag_assistant_returns_chunks_with_answer():
    assistant = RAGAssistant(
        embedding_service=MockEmbeddingService(),
        vector_store=MockVectorStore(),
        llm_service=MockLLMService(),
        formatter=MockFormatter(),
        context_builder=MockContextBuilder()
    )

    result = assistant.ask_question("What is PTO?")

    assert "answer" in result
    assert "chunks" in result