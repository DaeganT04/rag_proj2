import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from ingest import TextChunker, DocumentIngestor


def test_text_chunker_creates_chunks():
    chunker = TextChunker(chunk_size=10, overlap=2)

    chunks = chunker.chunk_text("abcdefghijklmnopqrstuvwxyz")

    assert len(chunks) > 1
    assert chunks[0] == "abcdefghij"


def test_text_chunker_rejects_bad_overlap():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, overlap=100)


def test_chunk_documents_adds_metadata():
    documents = [
        {
            "source": "doc1.txt",
            "text": "abcdefghijklmnopqrstuvwxyz"
        }
    ]

    chunker = TextChunker(chunk_size=10, overlap=2)
    chunked_docs = chunker.chunk_documents(documents)

    assert len(chunked_docs) > 1
    assert chunked_docs[0]["source"] == "doc1.txt"
    assert chunked_docs[0]["chunk_id"] == 0
    assert "text" in chunked_docs[0]


def test_document_ingestor_loads_txt_file(tmp_path):
    test_file = tmp_path / "sample.txt"
    test_file.write_text("This is a test document.", encoding="utf-8")

    ingestor = DocumentIngestor()
    documents = ingestor.ingest(tmp_path)

    assert len(documents) == 1
    assert documents[0]["source"] == "sample.txt"
    assert documents[0]["text"] == "This is a test document."


def test_document_ingestor_ignores_unsupported_files(tmp_path):
    test_file = tmp_path / "sample.csv"
    test_file.write_text("name,value", encoding="utf-8")

    ingestor = DocumentIngestor()
    documents = ingestor.ingest(tmp_path)

    assert documents == []
def test_text_chunker_handles_empty_text():
    chunker = TextChunker(chunk_size=10, overlap=2)

    chunks = chunker.chunk_text("")

    assert chunks == []


def test_text_chunker_handles_short_text():
    chunker = TextChunker(chunk_size=100, overlap=10)

    chunks = chunker.chunk_text("short text")

    assert chunks == ["short text"]


def test_text_chunker_strips_whitespace():
    chunker = TextChunker(chunk_size=20, overlap=5)

    chunks = chunker.chunk_text("     hello world     ")

    assert chunks[0] == "hello world"


def test_chunk_documents_handles_empty_document_list():
    chunker = TextChunker(chunk_size=10, overlap=2)

    chunked_docs = chunker.chunk_documents([])

    assert chunked_docs == []


def test_document_ingestor_ignores_empty_txt_file(tmp_path):
    test_file = tmp_path / "empty.txt"
    test_file.write_text("", encoding="utf-8")

    ingestor = DocumentIngestor()
    documents = ingestor.ingest(tmp_path)

    assert documents == []


def test_document_ingestor_loads_multiple_txt_files(tmp_path):
    file_one = tmp_path / "one.txt"
    file_two = tmp_path / "two.txt"

    file_one.write_text("First document.", encoding="utf-8")
    file_two.write_text("Second document.", encoding="utf-8")

    ingestor = DocumentIngestor()
    documents = ingestor.ingest(tmp_path)

    sources = [doc["source"] for doc in documents]

    assert len(documents) == 2
    assert "one.txt" in sources
    assert "two.txt" in sources


def test_document_ingestor_ignores_directories(tmp_path):
    folder = tmp_path / "nested_folder"
    folder.mkdir()

    ingestor = DocumentIngestor()
    documents = ingestor.ingest(tmp_path)

    assert documents == []