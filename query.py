import os
import logging
import warnings
from abc import ABC, abstractmethod

import chromadb
from sentence_transformers import SentenceTransformer
from ollama import chat

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

VECTOR_DB_FOLDER = "vector_db"
COLLECTION_NAME = "rag_collection"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3"


class EmbeddingService:
    def __init__(self, model_name=EMBED_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed(self, text):
        return self.model.encode(text).tolist()


class VectorStore:
    def __init__(self, db_path=VECTOR_DB_FOLDER, collection_name=COLLECTION_NAME):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name=collection_name)

    def search(self, query_embedding, n_results=3):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )


class LLMServiceInterface(ABC):
    @abstractmethod
    def generate(self, query_text, context):
        pass


class OllamaLLMService(LLMServiceInterface):
    def __init__(self, model_name=LLM_MODEL):
        self.model_name = model_name

    def generate(self, query_text, context):
        prompt = f"""
You are a retrieval-augmented assistant.

Answer the user's question using ONLY the provided context.
If the answer is not in the context, say: "I do not know based on the provided documents."

Be concise and accurate.
At the end, include a short citation line listing the source file names and chunk numbers you used.

Context:
{context}

Question:
{query_text}
"""

        response = chat(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You answer strictly from provided context and do not hallucinate."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.message.content


class ChunkFormatter:
    def format(self, results):
        formatted_chunks = []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        for doc, metadata in zip(documents, metadatas):
            formatted_chunks.append({
                "source": metadata["source"],
                "chunk_id": metadata["chunk_id"],
                "text": doc
            })

        return formatted_chunks


class ContextBuilder:
    def build(self, chunks):
        context_parts = []

        for chunk in chunks:
            context_parts.append(
                f"Source: {chunk['source']} | Chunk: {chunk['chunk_id']}\n"
                f"{chunk['text']}"
            )

        return "\n\n".join(context_parts)


class RAGAssistant:
    def __init__(self, embedding_service, vector_store, llm_service, formatter, context_builder):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.formatter = formatter
        self.context_builder = context_builder

    def ask_question(self, query_text, n_results=3):
        query_embedding = self.embedding_service.embed(query_text)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=n_results
        )

        chunks = self.formatter.format(results)

        if not chunks:
            return {
                "answer": "I do not know based on the provided documents.",
                "chunks": []
            }

        context = self.context_builder.build(chunks)
        answer = self.llm_service.generate(query_text, context)

        return {
            "answer": answer,
            "chunks": chunks
        }


def create_rag_assistant():
    embedding_service = EmbeddingService()
    vector_store = VectorStore()
    llm_service = OllamaLLMService()
    formatter = ChunkFormatter()
    context_builder = ContextBuilder()

    return RAGAssistant(
        embedding_service,
        vector_store,
        llm_service,
        formatter,
        context_builder
    )


def ask_question(query_text, n_results=3):
    assistant = create_rag_assistant()
    return assistant.ask_question(query_text, n_results)