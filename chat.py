import ollama
from retriever import retrieve_context
from config import LLM_MODEL

# Strict System Prompt to prevent hallucination
SYSTEM_PROMPT = """You are a helpful and precise AI assistant. 
Answer the user's question ONLY using the provided context. 
If the answer is not explicitly stated in the context, you MUST reply with exactly: "I don't have that information based on the provided documents."
Do not hallucinate, guess, or use outside knowledge. 
When providing an answer, always cite the source filename and page number in parentheses, e.g., (Source: filename.pdf, Page: 1)."""

def build_prompt(context_chunks: list, query: str) -> str:
    """Formats the context and query into a single prompt for the LLM."""
    context_text = ""
    for i, chunk in enumerate(context_chunks, 1):
        context_text += f"[Chunk {i}] Source: {chunk['source']}, Page: {chunk['page']}\nText: {chunk['text']}\n\n"
        
    return f"Context:\n{context_text}\n\nUser Question: {query}\n\nAnswer:"

def generate_response(query: str) -> dict:
    """
    Main chat function. Retrieves context, builds prompt, calls Qwen, and returns reply + sources.
    """
    # 1. Retrieve relevant context
    context_chunks = retrieve_context(query)
    
    # 2. Handle missing context
    if not context_chunks:
        return {
            "reply": "I don't have that information based on the provided documents.",
            "sources": []
        }
    
    # 3. Build the prompt
    prompt = build_prompt(context_chunks, query)
    
    # 4. Call Qwen via Ollama
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            options={
                "temperature": 0.1,  # Low temperature for factual, grounded responses
                "num_predict": 500   # Limit response length for <5s latency
            }
        )
        reply_text = response["message"]["content"].strip()
        
    except Exception as e:
        print(f"[!] Ollama API Error: {e}")
        return {
            "reply": "Error: Could not connect to the LLM. Please ensure Ollama is running.",
            "sources": []
        }
    
    # 5. Extract unique sources for the frontend/API to display
    sources = []
    for chunk in context_chunks:
        source_entry = f"{chunk['source']} (Page {chunk['page']})"
        if source_entry not in sources:
            sources.append(source_entry)
            
    return {
        "reply": reply_text,
        "sources": sources
    }

if __name__ == "__main__":
    # Quick test block to verify chat logic
    test_queries = [
        "What is the professional summary of the candidate?",
        "What is the capital of France?" # Should trigger the "I don't have that information" fallback
    ]
    
    for q in test_queries:
        print(f"\n[*] Query: '{q}'")
        result = generate_response(q)
        print(f"Reply: {result['reply']}")
        print(f"Sources: {result['sources']}")