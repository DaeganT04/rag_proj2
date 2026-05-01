## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system that enables a Large Language Model (LLM) to answer questions using a specific set of user-provided documents. Traditional LLMs rely solely on pre-trained knowledge and may lack access to private, domain-specific, or recently updated information. This system addresses that limitation by retrieving relevant information from local documents and using that information to generate grounded responses.

The assistant follows a full RAG pipeline: it loads documents, splits them into smaller chunks, converts those chunks into vector embeddings, stores them in a vector database, retrieves the most relevant sections based on a user query, and generates an answer using an LLM. The system is designed to ensure that the model answers strictly using the retrieved document context, reducing the risk of hallucinated or unsupported responses.

In Project 2, the system has been refactored to improve modularity, maintainability, and testability by applying core software design principles (SOLID), introducing a more structured architecture, and adding comprehensive testing.

---

## Use Case

The assistant functions as a document-based knowledge system capable of answering questions about a collection of internal documents. This type of system can be applied in several real-world scenarios, including:

- Company HR policy assistants  
- IT helpdesk documentation systems  
- Internal knowledge base tools  
- Technical manual search assistants  

A user submits a question, the system retrieves the most relevant document sections, and the LLM generates an answer grounded in that retrieved information.

---

## System Architecture

The system follows the standard Retrieval-Augmented Generation pipeline:

1. **Document Ingestion**
   - Documents are loaded from the `data/` folder
   - Text is extracted from `.txt` and `.pdf` files

2. **Chunking**
   - Documents are split into smaller overlapping chunks to preserve context and improve retrieval accuracy

3. **Embedding**
   - Each chunk is converted into a vector embedding using the Sentence Transformers model (`all-MiniLM-L6-v2`)

4. **Storage**
   - Embeddings, text, and metadata are stored in a ChromaDB vector database

5. **Query Processing**
   - The user’s question is converted into an embedding
   - Similarity search is performed against stored embeddings

6. **Retrieval**
   - The most relevant chunks are retrieved

7. **Response Generation**
   - Retrieved chunks are passed to the LLM (Llama3 via Ollama)
   - The model generates an answer using only the provided context

---

## Chunking Strategy

Documents are split into chunks of approximately 400 characters with an overlap of 100 characters between adjacent chunks. This approach ensures that:

- Each chunk contains enough context to be meaningful  
- Retrieval remains precise and relevant  
- Important information near chunk boundaries is preserved  

---

## Vector Database and Retrieval

The system uses **ChromaDB** as the vector database due to its efficient similarity search and support for persistent local storage. Each chunk’s embedding is stored alongside metadata, including:

- Source file name  
- Chunk ID  
- Chunk size  

When a user query is received, it is embedded using the same model and compared to stored vectors. The system retrieves the top three most relevant chunks, which are then used as context for answer generation.

---

## LLM Integration and Prompting

The retrieved chunks are passed to the LLM (Llama3 running locally via Ollama) along with the user’s question. A carefully designed prompt ensures that:

- The model answers **only using the provided context**  
- If the answer is not found, it responds accordingly  
- The response remains concise and accurate  
- Sources are cited when applicable  

This approach significantly reduces hallucination and ensures responses remain grounded in the document data.

---

## Running the Project

1. Install dependencies:

```bash
pip install -r requirements.txt

2. Run ollama

In new terminal run 
ollama serve 

then ollama pull llama3

3. Running the project

In original terminal start he vector database
python ingest.py

then run the program 
python main.py

## Docker Usage

Build the Docker image:

```bash
docker build -t rag-app .

Ollama serve 

docker run -it -e OLLAMA_HOST=http://host.docker.internal:11434 rag-app
