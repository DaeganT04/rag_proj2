import os
import logging
import warnings

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from pathlib import Path

import chromadb
from pypdf import PdfReader
from pypdf.errors import PdfReadError, PdfStreamError
from sentence_transformers import SentenceTransformer

DATA_FOLDER = "data"
VECTOR_DB_FOLDER = "vector_db"
COLLECTION_NAME = "rag_collection"


def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""

        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                text += f"\n[Page {page_num}]\n{page_text}\n"

        return text

    except (PdfReadError, PdfStreamError, Exception) as e:
        print(f"Skipping bad PDF: {file_path} ({e})")
        return ""


def ingest_documents(data_folder):
    documents = []

    for file_name in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file_name)

        if os.path.isfile(file_path):
            suffix = Path(file_name).suffix.lower()

            if suffix == ".txt":
                text = load_txt(file_path)
                if text.strip():
                    documents.append({
                        "source": file_name,
                        "text": text
                    })

            elif suffix == ".pdf":
                text = load_pdf(file_path)
                if text.strip():
                    documents.append({
                        "source": file_name,
                        "text": text
                    })

    return documents


def chunk_text(text, chunk_size=400, overlap=100):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


def chunk_documents(documents, chunk_size=400, overlap=100):
    chunked_docs = []

    for doc in documents:
        chunks = chunk_text(doc["text"], chunk_size, overlap)

        for i, chunk in enumerate(chunks):
            chunked_docs.append({
                "source": doc["source"],
                "chunk_id": i,
                "text": chunk
            })

    return chunked_docs


def build_vector_store(chunked_docs):
    client = chromadb.PersistentClient(path=VECTOR_DB_FOLDER)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    documents = [doc["text"] for doc in chunked_docs]
    embeddings = model.encode(documents).tolist()
    ids = [f'{doc["source"]}_chunk_{doc["chunk_id"]}' for doc in chunked_docs]
    metadatas = [
        {
            "source": doc["source"],
            "chunk_id": doc["chunk_id"],
            "chunk_size": len(doc["text"])
        }
        for doc in chunked_docs
    ]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"Stored {len(chunked_docs)} chunks in vector database.")


if __name__ == "__main__":
    docs = ingest_documents(DATA_FOLDER)

    print("Loaded documents:")
    for doc in docs:
        print(f"- {doc['source']} ({len(doc['text'])} characters)")

    chunked = chunk_documents(docs)
    print(f"\nCreated {len(chunked)} chunks total.\n")

    for c in chunked[:3]:
        print(f"Source: {c['source']}, Chunk: {c['chunk_id']}")
        print(c["text"][:300])
        print("-" * 50)

    build_vector_store(chunked)