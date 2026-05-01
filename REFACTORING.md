# Refactoring Report

## Overview

In Project 1, the RAG system was built as a functional prototype focused on producing correct outputs. However, the design had several issues, including tightly coupled components, limited modularity, and difficulty in testing.

For Project 2, the codebase was refactored to improve maintainability, flexibility, and testability by applying core SOLID principles. The updated design separates responsibilities across components and introduces abstraction to reduce coupling.

---

## SOLID Principles Implemented

### 1. Single Responsibility Principle (SRP)

**Definition:**
Each class should have only one reason to change.

**Implementation:**
In the original design, a single class handled multiple responsibilities such as document ingestion, vector storage, querying, and LLM interaction. This made the code harder to maintain and extend.

The refactored design separates these concerns into distinct classes:

* `DocumentLoader` → handles document ingestion
* `VectorStore` → manages embeddings and storage
* `Retriever` → retrieves relevant chunks
* `LLMService` → handles LLM interaction
* `RAGPipeline` → orchestrates the workflow

**Result:**
Each component now has a clearly defined role, making the system easier to debug, test, and modify independently.

---

### 2. Dependency Inversion Principle (DIP)

**Definition:**
High-level modules should not depend on low-level modules. Both should depend on abstractions.

**Implementation:**
Previously, the main pipeline directly depended on a specific LLM implementation. This created tight coupling and made testing difficult.

In the refactored design, the `RAGPipeline` depends on an abstract LLM interface rather than a concrete implementation. The LLM is injected into the pipeline:

```python
class RAGPipeline:
    def __init__(self, llm_service):
        self.llm = llm_service
```

This allows different implementations (real or mock) to be used interchangeably.

**Result:**

* Easier testing using mock LLMs
* Flexibility to switch models (e.g., OpenAI, local LLM, etc.)
* Reduced coupling between components

---

## Why These Changes Were Necessary

The original codebase had the following issues:

* Multiple responsibilities in a single class
* Tight coupling between components
* Difficult to test due to direct external dependencies

By applying SRP and DIP:

* The system became modular and easier to extend
* Components can be tested independently
* External services (like LLMs) can be mocked

---

## Before vs After

### Before (Simplified)

```python
class RAGSystem:
    def ingest(self): ...
    def query(self): ...
    def call_llm(self): ...
```

### After (Simplified)

```python
class DocumentLoader: ...
class VectorStore: ...
class Retriever: ...
class LLMService: ...

class RAGPipeline:
    def __init__(self, llm_service):
        self.llm = llm_service
```

---

## Conclusion

The refactored system is now significantly more maintainable, testable, and scalable. By applying SRP and DIP, the architecture better reflects real-world software engineering practices and prepares the system for future enhancements such as swapping models, adding new data sources, or extending functionality without modifying existing code.
