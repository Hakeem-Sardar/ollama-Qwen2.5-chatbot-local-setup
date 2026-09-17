import os
import time
import hashlib
import chromadb
import ollama
from pypdf import PdfReader
from config import (
    DATA_DIR, CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
)

# Approximate characters per token for lightweight chunking
CHARS_PER_TOKEN = 4

def load_documents():
    """Loads PDF, TXT, CSV, and MD files from the data directory."""
    documents = []
    print(f"[*] Scanning directory: {DATA_DIR}")

    for filename in os.listdir(DATA_DIR):
        filepath = DATA_DIR / filename
        if not filepath.is_file():
            continue
        try:
            docs = _read_file(filepath, filename)
            documents.extend(docs)
        except Exception as e:
            print(f"[!] Error reading {filename}: {e}")

    print(f"[+] Loaded {len(documents)} document pages.")
    return documents

def _read_file(filepath, filename):
    """Reads a single file and returns a list of document dicts."""
    documents = []

    if filename.endswith(".pdf"):
        reader = PdfReader(str(filepath))
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                documents.append({"text": text, "source": filename, "page": page_num})

    elif filename.endswith(".txt") or filename.endswith(".md"):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        if text.strip():
            documents.append({"text": text, "source": filename, "page": 1})

    elif filename.endswith(".csv"):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        if text.strip():
            documents.append({"text": text, "source": filename, "page": 1})

    return documents

def chunk_text(text, source, page):
    """Splits text into chunks based on character limits approximating token counts."""
    chunk_size_chars = CHUNK_SIZE * CHARS_PER_TOKEN
    overlap_chars = CHUNK_OVERLAP * CHARS_PER_TOKEN

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size_chars
        chunk_content = text[start:end].strip()

        if chunk_content:
            chunks.append({
                "text": chunk_content,
                "source": source,
                "page": page
            })

        start += (chunk_size_chars - overlap_chars)

    return chunks

def get_embeddings(texts):
    """Generates embeddings using Ollama."""
    print(f"[*] Generating embeddings for {len(texts)} chunks via {EMBEDDING_MODEL}...")
    embeddings = []
    for i, text in enumerate(texts):
        response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
        embeddings.append(response["embedding"])
        if (i + 1) % 10 == 0:
            print(f"    -> Embedded {i + 1}/{len(texts)} chunks")
    return embeddings

def store_in_chromadb(chunks, embeddings):
    """Stores chunks and embeddings into ChromaDB with metadata."""
    print(f"[*] Connecting to ChromaDB at {CHROMA_PERSIST_DIR}...")
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        content_hash = hashlib.md5(
            f"{chunk['source']}_{chunk['page']}_{chunk['text'][:100]}".encode()
        ).hexdigest()
        ids.append(content_hash)
        documents.append(chunk["text"])
        metadatas.append({
            "source": chunk["source"],
            "page": chunk["page"],
            "timestamp": time.time()
        })

    print(f"[*] Upserting {len(ids)} chunks into collection '{CHROMA_COLLECTION_NAME}'...")
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    print(f"[+] Successfully stored {len(ids)} chunks in ChromaDB.")

def ingest_single_file(filepath):
    """
    Ingests a single file into ChromaDB.
    Called by the Streamlit UI when a user uploads a document.
    Returns the number of chunks stored.
    """
    filename = os.path.basename(filepath)
    docs = _read_file(filepath, filename)

    if not docs:
        return 0

    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc["text"], doc["source"], doc["page"]))

    if not all_chunks:
        return 0

    texts_to_embed = [chunk["text"] for chunk in all_chunks]
    embeddings = get_embeddings(texts_to_embed)
    store_in_chromadb(all_chunks, embeddings)

    return len(all_chunks)

def main():
    print("=== Starting RAG Ingestion Pipeline ===")

    raw_docs = load_documents()
    if not raw_docs:
        print("[!] No documents found in data/ folder. Please add PDFs or TXTs and re-run.")
        return

    all_chunks = []
    for doc in raw_docs:
        all_chunks.extend(chunk_text(doc["text"], doc["source"], doc["page"]))
    print(f"[+] Created {len(all_chunks)} total chunks.")

    texts_to_embed = [chunk["text"] for chunk in all_chunks]
    embeddings = get_embeddings(texts_to_embed)

    store_in_chromadb(all_chunks, embeddings)

    print("=== Ingestion Complete ===")

if __name__ == "__main__":
    main()