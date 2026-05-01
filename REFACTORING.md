# Refactoring Report

## Overview

In Project 1, the Retrieval-Augmented Generation (RAG) system was implemented as a functional prototype focused on correctness and basic functionality. While the system worked, the internal design had several limitations:

- Large functions handled multiple responsibilities
- Tight coupling between components (LLM, vector database, embeddings)
- Limited flexibility for adding new features
- Difficult to test due to reliance on external services
- Code that was harder to read and maintain

In Project 2, the system was refactored to improve software quality by applying SOLID design principles. The main goals of this refactoring were:

- Improve modularity and separation of concerns
- Reduce coupling between components
- Make the system easier to extend and modify
- Enable proper unit testing using mocks
- Improve readability and maintainability

---

## SOLID Principles Implemented

This project implements three SOLID principles:

- Single Responsibility Principle (S)
- Open/Closed Principle (O)
- Dependency Inversion Principle (D)

---

## 1. Single Responsibility Principle (S)

### Description

The Single Responsibility Principle states that a class should have only one responsibility or reason to change.

### Problem in Original Code

In Project 1, core functionality was implemented inside large functions such as `ask_question()`. These functions performed multiple tasks:

- Generating embeddings
- Querying the vector database
- Formatting results
- Building context
- Calling the LLM

This caused several problems:

- Hard to understand and debug
- Difficult to test individual components
- Changes in one part could break other parts
- Poor separation of concerns

### Refactored Solution

The system was split into multiple classes, each handling a single responsibility:

- `EmbeddingService` → generates embeddings
- `VectorStore` → retrieves data from ChromaDB
- `OllamaLLMService` → handles LLM interaction
- `ChunkFormatter` → formats retrieved chunks
- `ContextBuilder` → builds the context string
- `RAGAssistant` → orchestrates the workflow

### Result

- Code is modular and easier to understand
- Each component can be tested independently
- Changes are isolated to specific classes
- Improved maintainability and readability

---

## 2. Open/Closed Principle (O)

### Description

The Open/Closed Principle states that software should be open for extension but closed for modification.

### Problem in Original Code

In Project 1, document ingestion used conditional logic:

```python
if suffix == ".txt":
    ...
elif suffix == ".pdf":
    ...