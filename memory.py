import sqlite3
import time
from config import SQLITE_DB_PATH

# Number of previous turns to keep in active memory for the LLM context
MEMORY_TURNS = 3 

def init_db():
    """Initializes the SQLite database for chat history."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            query TEXT,
            reply TEXT,
            sources TEXT,
            timestamp REAL
        )
    ''')
    conn.commit()
    conn.close()

def save_turn(session_id: str, query: str, reply: str, sources: list):
    """Saves a single Q&A turn to the SQLite database."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    
    # Convert sources list to a simple string for SQLite storage
    sources_str = " | ".join(sources) if sources else "None"
    
    cursor.execute('''
        INSERT INTO chat_history (session_id, query, reply, sources, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (session_id, query, reply, sources_str, time.time()))
    
    conn.commit()
    conn.close()

def get_recent_history(session_id: str) -> list:
    """Retrieves the last N turns of conversation for a given session."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT query, reply FROM chat_history 
        WHERE session_id = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (session_id, MEMORY_TURNS))
    
    rows = cursor.fetchall()
    conn.close()
    
    # Reverse to get chronological order (oldest of the recent turns first)
    return list(reversed(rows))

def format_history_for_prompt(history: list) -> str:
    """Formats the retrieved history into a string to prepend to the system prompt."""
    if not history:
        return ""
    
    history_text = "Previous Conversation:\n"
    for q, a in history:
        history_text += f"User: {q}\nAssistant: {a}\n"
    history_text += "\n"
    return history_text

# Initialize the database as soon as the module is imported
init_db()

if __name__ == "__main__":
    # Quick test block
    test_session = "test_session_123"
    
    print("[*] Saving test turn to memory...")
    save_turn(test_session, "Hello, who are you?", "I am a RAG chatbot.", ["doc.pdf"])
    
    print("[*] Retrieving recent history...")
    history = get_recent_history(test_session)
    print("Formatted History:\n", format_history_for_prompt(history))