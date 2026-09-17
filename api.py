from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Import the core chat logic we just built
from chat import generate_response

# Initialize FastAPI app
app = FastAPI(
    title="Qwen RAG Chatbot API",
    description="Backend API for the Qwen RAG Chatbot using Ollama and ChromaDB",
    version="1.0.0"
)

# --- Pydantic Models for Request/Response Validation ---

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None  # Optional: Frontend can provide this to maintain conversation state

class ChatResponse(BaseModel):
    reply: str
    sources: List[str]
    session_id: str

# --- Endpoints ---

@app.get("/health")
async def health_check():
    """Simple health check endpoint to verify the API is running."""
    return {"status": "ok", "service": "qwen-rag-api"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint.
    Receives a query (and optional session_id), processes it through the RAG pipeline,
    and returns the grounded reply, sources, and the session_id.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Call the core RAG logic from chat.py
        result = generate_response(query=request.query, session_id=request.session_id)
        
        return ChatResponse(
            reply=result["reply"],
            sources=result["sources"],
            session_id=result["session_id"]
        )
        
    except Exception as e:
        # Catch any unexpected errors (e.g., Ollama down, DB locked) and return a clean 500 error
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")