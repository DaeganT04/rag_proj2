import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from query import ChunkFormatter, ContextBuilder


def test_chunk_formatter_formats_results():
    results = {
        "documents": [["This is chunk text."]],
        "metadatas": [[{"source": "doc1.txt", "chunk_id": 2}]]
    }

    formatter = ChunkFormatter()
    chunks = formatter.format(results)

    assert len(chunks) == 1
    assert chunks[0]["source"] == "doc1.txt"
    assert chunks[0]["chunk_id"] == 2
    assert chunks[0]["text"] == "This is chunk text."


def test_context_builder_builds_context_string():
    chunks = [
        {
            "source": "doc1.txt",
            "chunk_id": 0,
            "text": "First chunk."
        },
        {
            "source": "doc2.txt",
            "chunk_id": 1,
            "text": "Second chunk."
        }
    ]

    builder = ContextBuilder()
    context = builder.build(chunks)

    assert "Source: doc1.txt | Chunk: 0" in context
    assert "First chunk." in context
    assert "Source: doc2.txt | Chunk: 1" in context
    assert "Second chunk." in context


def test_context_builder_handles_empty_chunks():
    builder = ContextBuilder()
    context = builder.build([])

    assert context == ""

def test_chunk_formatter_handles_multiple_chunks():
    results = {
        "documents": [["First text.", "Second text."]],
        "metadatas": [[
            {"source": "doc1.txt", "chunk_id": 0},
            {"source": "doc2.txt", "chunk_id": 1}
        ]]
    }

    formatter = ChunkFormatter()
    chunks = formatter.format(results)

    assert len(chunks) == 2
    assert chunks[0]["text"] == "First text."
    assert chunks[1]["text"] == "Second text."


def test_chunk_formatter_returns_empty_list_for_empty_results():
    results = {
        "documents": [[]],
        "metadatas": [[]]
    }

    formatter = ChunkFormatter()
    chunks = formatter.format(results)

    assert chunks == []


def test_context_builder_includes_chunk_separation():
    chunks = [
        {"source": "doc1.txt", "chunk_id": 0, "text": "First chunk."},
        {"source": "doc2.txt", "chunk_id": 1, "text": "Second chunk."}
    ]

    builder = ContextBuilder()
    context = builder.build(chunks)

    assert "\n\n" in context


def test_context_builder_preserves_chunk_text():
    chunks = [
        {
            "source": "policy.txt",
            "chunk_id": 7,
            "text": "Exact policy language should stay the same."
        }
    ]

    builder = ContextBuilder()
    context = builder.build(chunks)

    assert "Exact policy language should stay the same." in context