import os
import logging
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import chromadb
from pypdf import PdfReader
from pypdf.errors import PdfReadError, PdfStreamError
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

DATA_FOLDER = "data"
VECTOR_DB_FOLDER = "vector_db"
COLLECTION_NAME = "rag_collection"
EMBED_MODEL = "all-MiniLM-L6-v2"


class DocumentLoader(ABC):
    @abstractmethod
    def load(self, file_path):
        pass


class TxtLoader(DocumentLoader):
    def load(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()


class PdfLoader(DocumentLoader):
    def load(self, file_path):
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


class DocumentIngestor:
    def __init__(self):
        self.loaders = {
            ".txt": TxtLoader(),
            ".pdf": PdfLoader()
        }

    def ingest(self, data_folder):
        documents = []

        for file_name in os.listdir(data_folder):
            file_path = os.path.join(data_folder, file_name)

            if os.path.isfile(file_path):
                suffix = Path(file_name).suffix.lower()
                loader = self.loaders.get(suffix)

                if loader:
                    text = loader.load(file_path)

                    if text.strip():
                        documents.append({
                            "source": file_name,
                            "text": text
                        })

        return documents


class TextChunker:
    def __init__(self, chunk_size=400, overlap=100):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text):
        chunks = []
        start = 0
        step = self.chunk_size - self.overlap

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            start += step

        return chunks

    def chunk_documents(self, documents):
        chunked_docs = []

        for doc in documents:
            chunks = self.chunk_text(doc["text"])

            for i, chunk in enumerate(chunks):
                chunked_docs.append({
                    "source": doc["source"],
                    "chunk_id": i,
                    "text": chunk
                })

        return chunked_docs


class EmbeddingService:
    def __init__(self, model_name=EMBED_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents):
        return self.model.encode(documents).tolist()


class VectorStoreBuilder:
    def __init__(self, db_path=VECTOR_DB_FOLDER, collection_name=COLLECTION_NAME):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection_name = collection_name

    def reset_collection(self):
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

        return self.client.get_or_create_collection(name=self.collection_name)

    def build(self, chunked_docs, embedding_service):
        collection = self.reset_collection()

        documents = [doc["text"] for doc in chunked_docs]
        embeddings = embedding_service.embed_documents(documents)

        ids = [
            f'{doc["source"]}_chunk_{doc["chunk_id"]}'
            for doc in chunked_docs
        ]

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


class IngestionPipeline:
    def __init__(self, ingestor, chunker, embedding_service, vector_store_builder):
        self.ingestor = ingestor
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.vector_store_builder = vector_store_builder

    def run(self, data_folder):
        docs = self.ingestor.ingest(data_folder)

        print("Loaded documents:")
        for doc in docs:
            print(f"- {doc['source']} ({len(doc['text'])} characters)")

        chunked_docs = self.chunker.chunk_documents(docs)
        print(f"\nCreated {len(chunked_docs)} chunks total.\n")

        for chunk in chunked_docs[:3]:
            print(f"Source: {chunk['source']}, Chunk: {chunk['chunk_id']}")
            print(chunk["text"][:300])
            print("-" * 50)

        self.vector_store_builder.build(
            chunked_docs,
            self.embedding_service
        )


def main():
    pipeline = IngestionPipeline(
        ingestor=DocumentIngestor(),
        chunker=TextChunker(chunk_size=400, overlap=100),
        embedding_service=EmbeddingService(),
        vector_store_builder=VectorStoreBuilder()
    )

    pipeline.run(DATA_FOLDER)


if __name__ == "__main__":
    main()