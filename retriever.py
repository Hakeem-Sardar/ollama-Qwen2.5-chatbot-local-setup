import chromadb
import ollama
from config import (
    CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, 
    EMBEDDING_MODEL, TOP_K
)

def embed_query(query: str) -> list:
    """Generates an embedding for the user's query using Ollama."""
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=query)
    return response["embedding"]

def retrieve_context(query: str) -> list:
    """
    Retrieves the top K most relevant chunks from ChromaDB for a given query.
    Returns a list of dictionaries containing the text and metadata (source, page).
    """
    # 1. Embed the query
    query_embedding = embed_query(query)
    
    # 2. Connect to ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
    
    # 3. Perform Vector Search for Top-K chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas"]
    )
    
    # 4. Format the results
    context_chunks = []
    
    # ChromaDB returns lists of lists, so we extract the first (and only) query result
    if results and results['documents']:
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        for doc, meta in zip(documents, metadatas):
            context_chunks.append({
                "text": doc,
                "source": meta.get("source", "Unknown"),
                "page": meta.get("page", "N/A")
            })
            
    return context_chunks

if __name__ == "__main__":
    # Quick test block to verify retrieval works
    test_query = "What does the Qwen RAG chatbot use for vector storage?"
    print(f"[*] Testing retrieval for query: '{test_query}'")
    
    retrieved_chunks = retrieve_context(test_query)
    
    if not retrieved_chunks:
        print("[!] No context found. Did you run ingest.py first?")
    else:
        print(f"[+] Retrieved {len(retrieved_chunks)} chunks:\n")
        for i, chunk in enumerate(retrieved_chunks, 1):
            print(f"--- Chunk {i} ---")
            print(f"Source: {chunk['source']} | Page: {chunk['page']}")
            print(f"Text: {chunk['text'][:150]}...\n")