# AI Academic Supervisor

A Streamlit-based Retrieval-Augmented Generation (RAG) assistant that behaves like an academic supervisor for Industrial Economics course material.

The system answers questions using only pre-indexed lecture PDFs, then generates a grounded response with source-aware context.

## Project Overview

This project turns lecture slides and course documents into a searchable knowledge base, then connects that retrieval layer to a lightweight chat interface.

At runtime, the app does the following:
- Accepts a user question in a Streamlit chat UI.
- Embeds the question with OpenAI embeddings.
- Retrieves the most relevant chunks from a FAISS index.
- Builds a grounded prompt from retrieved chunks.
- Generates an answer with an OpenAI chat model.
- Displays source chunks used for transparency.

The result is a domain-focused assistant optimized for course support, revision, and exam preparation.

## Why RAG?

RAG is used here to make responses reliable and tied to real course material.

Without retrieval, a language model may:
- Hallucinate details.
- Miss course-specific definitions.
- Produce generic explanations not aligned with your module.

With retrieval, the system:
- Grounds answers in lecture content.
- Improves factual consistency.
- Makes answers auditable through source display.
- Allows easy content updates by re-indexing documents instead of retraining a model.

## System Architecture

### High-level Flow

1. Ingestion
- Load PDFs from data/raw.
- Extract page text.

2. Preprocessing
- Split documents into overlapping chunks (default: chunk_size=800, overlap=150).
- Preserve source and chunk IDs.

3. Embedding + Indexing
- Create embeddings with text-embedding-3-small.
- Store vectors in FAISS IndexFlatL2.
- Save metadata separately (chunk_id, source, text).

4. Query-time Retrieval
- Embed user question.
- Retrieve top-k chunks from FAISS.

5. Prompt Construction + Generation
- Build grounded prompt from retrieved chunks.
- Generate answer with gpt-4o-mini.

6. UI + Traceability
- Show response in Streamlit chat.
- Show chunk-level sources and distances in an expandable panel.

### Repository Map

- app.py: Streamlit chat application.
- build_index.py: End-to-end retrieval + generation smoke script.
- src/loaders.py: PDF loading and text extraction.
- src/chunking.py: Chunking logic with overlap.
- src/embedding.py: OpenAI embedding client and helpers.
- src/retrieval.py: FAISS index build/load/search + metadata persistence.
- src/prompting.py: Grounded prompt template construction.
- src/llm.py: OpenAI chat completion call.
- tests/run_retrieval.py: Build/rebuild retrieval assets.
- tests/query_search.py: Retrieval quality sanity check.
- tests/vectordb.py: Vector DB integrity check.

## Features

- Streamlit conversational interface with session chat history.
- Grounded answering from pre-indexed course material.
- Source chunk transparency in UI.
- Modular pipeline (load, chunk, embed, retrieve, prompt, generate).
- FAISS-backed local vector search.
- Lightweight scripts to build and validate index artifacts.
- Conda and pip environment options.

## Tech Stack

### Core
- Python 3.11
- Streamlit
- OpenAI API
- FAISS (faiss-cpu)
- NumPy
- PyPDF
- python-dotenv

### Supporting
- langchain / langchain-community (installed dependencies)
- pandas
- tiktoken

## Design Decisions

### 1. Use RAG instead of model fine-tuning
Reason:
- Faster iteration for changing lecture material.
- Lower cost and complexity.
- Better explainability via retrieved chunks.

Trade-off:
- Retrieval quality strongly affects final answer quality.

### 2. Keep vector store local with FAISS
Reason:
- Simple setup and fast similarity search.
- Good fit for single-user or small-course workflows.

Trade-off:
- Not optimized for distributed production workloads.

### 3. Separate vectors and metadata
Reason:
- Efficient similarity search over raw vectors.
- Clean retrieval output assembly from metadata.pkl.

Trade-off:
- Requires keeping index and metadata files in sync.

### 4. Use overlapping fixed-size chunks
Reason:
- Preserves context continuity between chunk boundaries.
- Keeps embedding generation straightforward.

Trade-off:
- Character-based chunking may not align perfectly with semantic boundaries.

### 5. Constrain prompt to provided context
Reason:
- Reduces hallucinations.
- Keeps behavior aligned with course-authoritative content.

Trade-off:
- System intentionally refuses to answer beyond indexed material.

### 6. Expose sources in the UI
Reason:
- Builds user trust.
- Supports verification and study workflows.

Trade-off:
- Adds visual complexity versus a plain chat interface.

## Data and Artifacts

- Raw PDFs: data/raw
- Optional preprocessed outputs: data/preprocessed
- Vector index artifacts: vectorstore/faiss.index, vectorstore/metadata.pkl, vectorstore/chunks_metadata.pkl

Note: data and vectorstore are ignored by git in this repository configuration.

## Setup

### Option A: Conda

```bash
conda env create -f environment.yml
conda activate vchat
```

### Option B: pip

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a .env file at project root:

```env
OPENAI_API_KEY=your_key_here
```

## Build or Rebuild the Index

```bash
python tests/run_retrieval.py
```

This script loads PDFs, chunks text, embeds chunks, builds FAISS index, and writes metadata.

## Run the App

```bash
streamlit run app.py
```

## Quick Validation Scripts

```bash
python tests/vectordb.py
python tests/query_search.py
python build_index.py
```

## Current Limitations

- Retrieval uses L2 distance with IndexFlatL2 and no reranking.
- Chunking is character-based, not semantic/token-based.
- No automated unit/integration test suite yet; current checks are script-based.
- Single-domain scope: tuned for indexed Industrial Economics material.

## Future Improvements

- Add semantic or token-aware chunking.
- Add reranking stage for better top-k precision.
- Add evaluation set with retrieval and answer quality metrics.
- Add citation formatting in generated answer text.
- Add structured configuration for model/index/chunk parameters.
