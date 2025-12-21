# Notebook Explanations

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
