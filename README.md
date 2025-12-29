# Notebook Explanations

---

# Backend Explanation

## Architecture
FastAPI backend with LangGraph for orchestrating the ingestion pipeline.

## Ingestion Flow
`POST /api/v1/ingestion/ingest` → Uploads PDF and processes it through:

1. **Load** → Parse PDF using `PyMuPDF4LLM` (Markdown extraction)
2. **Chunk** → Split into chunks (max 1500 chars) using `RecursiveCharacterTextSplitter`
3. **Contextual Retrieval** → Use Google Gemini with caching to generate context for each chunk
4. **Embed** → Generate vectors using Azure OpenAI `text-embedding-3-large`
5. **Store** → Save to Weaviate vector database

## Key Components
| Component | Purpose |
|-----------|---------|
| `backend/main.py` | FastAPI app entry |
| `backend/api/v1/ingestion.py` | Ingestion endpoint |
| `backend/graphs/ingestion/graph.py` | LangGraph pipeline |
| `backend/graphs/ingestion/nodes/` | Individual processing nodes |
| `backend/services/ingestion/` | Loader, VectorStore services |
| `backend/core/db/weaviate_client.py` | Weaviate connection manager |

## External Services
- **Weaviate** (Docker) → Vector DB
- **Google Gemini** → Context generation with caching
- **Azure OpenAI** → Embeddings

---

## RUN BACKEND CODE WITH 
```
uv run uvicorn backend.main:app --reload
```

### or directly run thorugh docker (preferred: as this setups redis and weviate as well )
```
docker compose up -d --build
```


## Overview
This is a RAG (Retrieval Augmented Generation) system for the Nepal Constitution document. There are 2 main notebooks:

---

## 1. `contextual_retrival.ipynb`
**What it does:** Prepares the document for search

**Steps:**
1. **Load PDF** → Read the Nepal Constitution PDF
2. **Split into chunks** → Break document into small pieces (max 1500 characters)
3. **Generate context** → For each chunk, use Google Gemini to create a description explaining where it fits in the document
4. **Create embeddings** → Convert each chunk to vectors using Azure OpenAI
5. **Upload to Weaviate** → Store chunks + context + vectors in database

**Output:** 
- All chunks are now searchable in Weaviate with their embeddings and context

---

## 2. `retrival.ipynb`
**What it does:** Search the document and retrieve relevant results

**Main components:**

### Individual Search Methods:
- **Vector Search** → Find chunks similar in meaning to your query (semantic search)
- **BM25 Search** → Find chunks that have exact keywords from your query (keyword search)

### Hybrid Retrieval (Best Method):
1. **Get 25 results from BM25** (keyword matches)
2. **Get 25 results from Vector search** (semantic matches)
3. **Remove duplicates** (if same chunk appears in both, keep only one)
4. **Combine with RRF** (Reciprocal Rank Fusion) → Mathematically merge the two rankings
5. **Return Top 10** → Final ranked results

**Why hybrid?**
- Vector search alone might miss exact keywords
- BM25 alone might miss semantic meaning
- Combining both = best of both worlds

---

