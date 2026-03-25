# Mini RAG Application

A simple Retrieval-Augmented Generation (RAG) application built with Streamlit, LangChain, and Ollama for local document querying.

## Models Used

### Embedding Model: `nomic-embed-text`

- **Why chosen**: Open-source embedding model optimized for text similarity tasks. It provides high-quality embeddings while being lightweight for running locally. 

### LLM: `gemma2:2b`

- **Why chosen**: A light-weight, open-source llm from Google. The 2B parameter version allows for local execution on modest hardware. Provides relevent responses for RAG systems. Great from security stand point.

## Document Chunking Implementation

The document processing pipeline uses a two-stage chunking strategy:

1. **Markdown Header Splitting**: Documents are first split using `MarkdownHeaderTextSplitter` with the following header levels:
   
   - `#` (Header 1)
   - `##` (Header 2)
   - `###` (Header 3)
   - `####` (Header 4)
   
   This preserves document structure and ensures that semantically related content stays together.

2. **Recursive Character Splitting**: After header-based splitting, chunks are further divided using `RecursiveCharacterTextSplitter` with:
   
   - **Chunk size**: 1000 characters
   - **Overlap**: 200 characters
   
   The overlap ensures continuity between chunks, preventing loss of context at split points.

## Retrieval Implementation

- **Vector Store**: ChromaDB is used as the vector database for storing document embeddings
- **Retriever**: The Chroma vectorstore's built-in retriever is used to find the most relevant document chunks
- **Similarity Search**: Cosine similarity is used to match query embeddings with stored document embeddings

## Grounding to Retrieved Context

The system enforces grounding through a carefully designed prompt template that:

1. **Explicit Instructions**: The prompt clearly states: "Answer the following question based only on the provided context."

2. **Fallback Responses**: If sufficient context is not available, the model is instructed to respond with exactly: "I need more context to answer this question. Please provide additional information."

3. **Unrelated Questions**: For questions completely unrelated to the documents, the model responds with: "I cannot answer this question based on the provided documents."

4. **Temperature Setting**: The LLM uses a temperature of 0.7, providing some creativity while maintaining factual grounding.

This approach ensures that all responses are directly supported by the retrieved document content, preventing hallucination and maintaining reliability.

## Full Local Setup (LLM + RAG)

### Prerequisites

- Python 3.10+ installed
- Ollama installed and running (https://ollama.com/docs)

### 1. Clone and enter repository

```bash
git clone https://github.com/Sri170303/mini_rag
cd mini_rag
```

### 2. Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install streamlit langchain langchain-ollama langchain-community langchain-text-splitters chromadb requests
```

### 4. Pull required Ollama models

```bash
ollama pull nomic-embed-text
ollama pull gemma2:2b
```

> If you use other models, update `config.py` constants `EMBEDDING_MODEL` and `LLM_MODEL`.

### 5. Verify `config.py` settings (optional)

- `DATA_DIR = "data"`
- `CHROMA_DB_DIR = "chroma_db"`
- `EMBEDDING_MODEL = "nomic-embed-text"`
- `LLM_MODEL = "gemma2:2b"`

### 6. Run Streamlit app

```bash
streamlit run app.py
```

### 7. Open app in browser

- Visit `http://localhost:8501`

### 8. Initialize app flow

1. Use sidebar to fetch markdown from Google Drive or add `.md` files to `data/`
2. Click “Process Documents and Create Vector Store” to build `chroma_db`
3. Ask questions in the main input and observe retrieved context + answer
4. <mark>Since local LLM is used, the response time initially will be a bit higher for a session.</mark> <mark>Using the same session will reduce response time in further questions.</mark>

### 9. Check for common issues

- Ensure `ollama` daemon is running (`ollama run ...` may not be needed for these endpoints)
- Ensure `data/` exists and contains `.md` docs before processing
- If vector store does not exist, press process button again
- If model load fails, confirm pulled models and versions are available
