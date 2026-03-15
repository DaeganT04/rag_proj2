import os
import logging
import warnings

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import chromadb
from sentence_transformers import SentenceTransformer
from ollama import chat

VECTOR_DB_FOLDER = "vector_db"
COLLECTION_NAME = "rag_collection"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3"


def load_vector_store():
    client = chromadb.PersistentClient(path=VECTOR_DB_FOLDER)
    collection = client.get_collection(name=COLLECTION_NAME)
    return collection


def retrieve_chunks(query_text, n_results=3):
    collection = load_vector_store()
    model = SentenceTransformer(EMBED_MODEL)

    query_embedding = model.encode(query_text).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results


def format_results(results):
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


def build_context(chunks):
    context_parts = []

    for chunk in chunks:
        context_parts.append(
            f"Source: {chunk['source']} | Chunk: {chunk['chunk_id']}\n"
            f"{chunk['text']}"
        )

    return "\n\n".join(context_parts)


def generate_answer(query_text, chunks):
    context = build_context(chunks)

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
        model=LLM_MODEL,
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


def ask_question(query_text, n_results=3):
    results = retrieve_chunks(query_text, n_results=n_results)
    chunks = format_results(results)

    if not chunks:
        return {
            "answer": "I do not know based on the provided documents.",
            "chunks": []
        }

    answer = generate_answer(query_text, chunks)

    return {
        "answer": answer,
        "chunks": chunks
    }

if __name__ == "__main__":
    while True:
        user_query = input("Enter your question (or type quit): ").strip()

        if user_query.lower() == "quit":
            break

        result = ask_question(user_query, n_results=3)

        print("\nAnswer:\n")
        print(result["answer"])
        print("\nRetrieved Chunks:\n")

        for chunk in result["chunks"]:
            print(f"Source: {chunk['source']} | Chunk: {chunk['chunk_id']}")
            print(chunk["text"])
            print("-" * 60)