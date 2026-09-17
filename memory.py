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


def get_latest_session_id() -> str:
    """Retrieves the most recently active session ID from the database."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT session_id FROM chat_history ORDER BY timestamp DESC LIMIT 1')
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

def get_full_history(session_id: str) -> list:
    """Retrieves the complete chat history for a specific session, ordered chronologically."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT query, reply, sources FROM chat_history 
        WHERE session_id = ? 
        ORDER BY timestamp ASC
    ''', (session_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_all_sessions() -> list:
    """
    Retrieves all unique session IDs with their first message and last activity timestamp.
    Returns list of dicts: [{'session_id': '...', 'first_query': '...', 'last_active': timestamp}]
    """
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    
    # Get all unique sessions with their first query and most recent timestamp
    cursor.execute('''
        SELECT 
            session_id,
            (SELECT query FROM chat_history h2 WHERE h2.session_id = h1.session_id ORDER BY timestamp ASC LIMIT 1) as first_query,
            MAX(timestamp) as last_active,
            COUNT(*) as message_count
        FROM chat_history h1
        GROUP BY session_id
        ORDER BY last_active DESC
    ''')
    
    rows = cursor.fetchall()
    conn.close()
    
    sessions = []
    for session_id, first_query, last_active, message_count in rows:
        sessions.append({
            "session_id": session_id,
            "first_query": first_query or "Empty session",
            "last_active": last_active,
            "message_count": message_count
        })
    
    return sessions
    
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