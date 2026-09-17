from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict

from chat import generate_response
from memory import get_latest_session_id, get_full_history, get_all_sessions# <-- Added import

app = FastAPI(
    title="Qwen RAG Chatbot API",
    description="Backend API for the Qwen RAG Chatbot using Ollama and ChromaDB",
    version="1.0.0"
)

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    sources: List[str]
    session_id: str

class HistoryResponse(BaseModel):
    session_id: str
    messages: List[Dict]

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "qwen-rag-api"}

# --- NEW ENDPOINT ---
@app.get("/history", response_model=HistoryResponse)
async def get_history(session_id: Optional[str] = None):
    """
    Fetches chat history. If no session_id is provided, it fetches the most recent session.
    """
    if not session_id:
        session_id = get_latest_session_id()
        
    if not session_id:
        return {"session_id": "", "messages": []}

    rows = get_full_history(session_id)
    messages = []
    
    for q, r, s in rows:
        # Reconstruct the messages array for the frontend
        messages.append({"role": "user", "content": q})
        
        # Parse the sources string back into a list
        sources_list = s.split(" | ") if s and s != "None" else []
        messages.append({"role": "assistant", "content": r, "sources": sources_list})

    return {"session_id": session_id, "messages": messages}

@app.get("/sessions", response_model=List[Dict])
async def list_sessions():
    """Returns a list of all chat sessions with metadata"""
    return get_all_sessions()
    
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        result = generate_response(query=request.query, session_id=request.session_id)
        return ChatResponse(
            reply=result["reply"],
            sources=result["sources"],
            session_id=result["session_id"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")