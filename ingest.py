import os
import time
import hashlib
import chromadb
import ollama
from pypdf import PdfReader
from config import (
    DATA_DIR, CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME,
    LLM_MODEL, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
)

# Approximate characters per token for lightweight chunking (avoids heavy tokenizer dependencies)
CHARS_PER_TOKEN = 4 

def load_documents():
    """Loads PDF and TXT files from the data directory."""
    documents = []
    print(f"[*] Scanning directory: {DATA_DIR}")
    
    for filename in os.listdir(DATA_DIR):
        filepath = DATA_DIR / filename
        if not filepath.is_file():
            continue
            
        try:
            if filename.endswith(".pdf"):
                reader = PdfReader(str(filepath))
                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text()
                    if text.strip():
                        documents.append({"text": text, "source": filename, "page": page_num})
                        
            elif filename.endswith(".txt"):
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                if text.strip():
                    documents.append({"text": text, "source": filename, "page": 1})
                    
        except Exception as e:
            print(f"[!] Error reading {filename}: {e}")
            
    print(f"[+] Loaded {len(documents)} document pages.")
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
        # ollama.embeddings returns a dictionary with the 'embedding' key
        response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
        embeddings.append(response["embedding"])
        if (i + 1) % 10 == 0:
            print(f"    -> Embedded {i + 1}/{len(texts)} chunks")
    return embeddings

def store_in_chromadb(chunks, embeddings):
    """Stores chunks and embeddings into ChromaDB with metadata."""
    print(f"[*] Connecting to ChromaDB at {CHROMA_PERSIST_DIR}...")
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    
    # Get or create the collection
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
    
    ids = []
    documents = []
    metadatas = []
    
    for chunk in chunks:
        # Create a deterministic ID based on content and source to prevent duplicates on re-runs
        content_hash = hashlib.md5(f"{chunk['source']}_{chunk['page']}_{chunk['text'][:100]}".encode()).hexdigest()
        ids.append(content_hash)
        documents.append(chunk["text"])
        metadatas.append({
            "source": chunk["source"],
            "page": chunk["page"],
            "timestamp": time.time()
        })
        
    print(f"[*] Upserting {len(ids)} chunks into collection '{CHROMA_COLLECTION_NAME}'...")
    # Using upsert ensures that re-running the script updates existing chunks without throwing duplicate ID errors
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    print(f"[+] Successfully stored {len(ids)} chunks in ChromaDB.")

def main():
    print("=== Starting RAG Ingestion Pipeline ===")
    
    # 1. Load raw documents
    raw_docs = load_documents()
    if not raw_docs:
        print("[!] No documents found in data/ folder. Please add PDFs or TXTs and re-run.")
        return
        
    # 2. Chunk documents
    all_chunks = []
    for doc in raw_docs:
        all_chunks.extend(chunk_text(doc["text"], doc["source"], doc["page"]))
    print(f"[+] Created {len(all_chunks)} total chunks.")
    
    # 3. Embed chunks
    texts_to_embed = [chunk["text"] for chunk in all_chunks]
    embeddings = get_embeddings(texts_to_embed)
    
    # 4. Store in Vector DB
    store_in_chromadb(all_chunks, embeddings)
    
    print("=== Ingestion Complete ===")

if __name__ == "__main__":
    main()