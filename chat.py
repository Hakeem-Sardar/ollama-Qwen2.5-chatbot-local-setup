import uuid
import ollama
from retriever import retrieve_context
from memory import get_recent_history, format_history_for_prompt, save_turn
from config import LLM_MODEL

# Strict System Prompt to prevent hallucination
SYSTEM_PROMPT = """You are a helpful and precise AI assistant. 
Answer the user's question ONLY using the provided context and previous conversation history. 
If the answer is not explicitly stated in the context, you MUST reply with exactly: "I don't have that information based on the provided documents."
Do not hallucinate, guess, or use outside knowledge. 
When providing an answer, always cite the source filename and page number in parentheses, e.g., (Source: filename.pdf, Page: 1)."""

def build_prompt(context_chunks: list, query: str, history_text: str) -> str:
    """Formats the history, context, and query into a single prompt for the LLM."""
    context_text = ""
    for i, chunk in enumerate(context_chunks, 1):
        context_text += f"[Chunk {i}] Source: {chunk['source']}, Page: {chunk['page']}\nText: {chunk['text']}\n\n"
        
    return f"{history_text}Context:\n{context_text}\n\nUser Question: {query}\n\nAnswer:"

def generate_response(query: str, session_id: str = None) -> dict:
    """
    Main chat function. Retrieves history, retrieves context, builds prompt, calls Qwen, saves to memory, and returns reply + sources.
    """
    # Generate a session ID if none is provided (for single-turn API calls)
    if not session_id:
        session_id = str(uuid.uuid4())
        
    # 1. Retrieve recent conversation history
    history = get_recent_history(session_id)
    history_text = format_history_for_prompt(history)
    
    # 2. Retrieve relevant context for the current query
    context_chunks = retrieve_context(query)
    
    # 3. Handle missing context
    if not context_chunks:
        reply_text = "I don't have that information based on the provided documents."
        sources = []
    else:
        # 4. Build the prompt
        prompt = build_prompt(context_chunks, query, history_text)
        
        # 5. Call Qwen via Ollama
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
            reply_text = "Error: Could not connect to the LLM. Please ensure Ollama is running."
            sources = []
            
        # 6. Extract unique sources for the frontend/API to display
        sources = []
        for chunk in context_chunks:
            source_entry = f"{chunk['source']} (Page {chunk['page']})"
            if source_entry not in sources:
                sources.append(source_entry)
                
    # 7. Save the turn to SQLite memory
    save_turn(session_id, query, reply_text, sources)
            
    return {
        "reply": reply_text,
        "sources": sources,
        "session_id": session_id
    }

if __name__ == "__main__":
    # Quick test block to verify chat logic with memory
    test_session = "memory_test_session_999"
    
    test_queries = [
        "What is the professional summary of the candidate?",
        "Can you summarize that again briefly?" # This tests if memory is working
    ]
    
    print(f"[*] Starting test session: {test_session}\n")
    for q in test_queries:
        print(f"--- User: {q} ---")
        result = generate_response(q, session_id=test_session)
        print(f"Assistant: {result['reply']}")
        print(f"Sources: {result['sources']}\n")