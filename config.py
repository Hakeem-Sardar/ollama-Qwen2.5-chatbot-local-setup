import os
from pathlib import Path

# --- Directory Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "db"
LOGS_DIR = BASE_DIR / "logs"

# Ensure core directories exist
DATA_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# --- Ollama & Model Configuration ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5")         # 7B default
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# --- Chunking Configuration ---
CHUNK_SIZE = 500       # Target tokens per chunk
CHUNK_OVERLAP = 100    # Overlap to preserve context across chunks

# --- Retrieval Configuration ---
TOP_K = 4              # Number of top chunks to retrieve per query

# --- ChromaDB Configuration ---
CHROMA_COLLECTION_NAME = "qwen_rag_docs"
CHROMA_PERSIST_DIR = str(DB_DIR / "chroma_db")

# --- SQLite Metadata Configuration ---
SQLITE_DB_PATH = str(LOGS_DIR / "qa_history.db")