# Qwen RAG Chatbot — Project Blueprint

## Goal
Build an AI chatbot using Qwen that answers ONLY from a custom database (RAG). No hallucination.

## Stack
- LLM: Qwen2.5 (7B/14B) via Ollama
- Embeddings: nomic-embed-text via Ollama
- Vector DB: ChromaDB (persistent)
- Backend: Python + FastAPI
- Frontend: Streamlit
- Metadata: SQLite

## Architecture
User Query → Embed → Vector Search → Top-K Chunks → Prompt(System+Context+Query) → Qwen → Answer

## Folder Structure
qwen-chatbot/
├── data/                 # source docs (pdf, txt, csv, md)
├── db/                   # chromadb storage
├── logs/                 # qa_history.db
├── ingest.py             # chunk + embed + store
├── retriever.py          # query embedding + top-k search
├── chat.py               # prompt builder + Qwen call
├── memory.py             # conversation history
├── app.py                # streamlit UI
├── api.py                # fastapi endpoints
├── config.py             # constants
├── requirements.txt
└── README.md

## Requirements

### Functional
1. Ingest documents from `data/` into vector DB
2. Chunk size 500–800 tokens, overlap 100
3. Embed with `nomic-embed-text`
4. Store chunks + metadata (source, page, timestamp) in ChromaDB
5. Retrieve top 3–5 chunks per query
6. Pass context + query to Qwen2.5
7. Return grounded answer with source citations
8. Return "I don't have that information" if context missing
9. Maintain conversation memory (last N turns)
10. Log every Q&A pair to SQLite for future fine-tuning
11. Streamlit UI: chat input, history, sources toggle, upload docs sidebar
12. FastAPI: POST /chat → {reply, sources}

### Non-Functional
- No hallucination — strict system prompt
- Response latency < 5s (7B model)
- Re-run ingest.py to update DB (versioned)
- Persistent storage across restarts
- Configurable model, top_k, chunk_size via config.py

## System Prompt (STRICT)