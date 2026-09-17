# Qwen RAG Chatbot

A local, dependency-light RAG (Retrieval-Augmented Generation) chatbot powered by Qwen2.5, ChromaDB, and Ollama. This chatbot answers questions **strictly from your documents** with zero hallucination.

## ✨ Features

- **Grounded Responses**: Answers ONLY from your uploaded documents with source citations
- **Multiple Model Support**: Switch between Qwen2.5, Llama 3.1, Mistral, and more
- **Conversation Memory**: Maintains chat history across sessions
- **Document Upload**: Upload PDF, TXT, CSV, and MD files via UI
- **Source Citations**: Every answer includes references to source documents
- **Anti-Hallucination**: Returns "I don't have that information" when context is missing
- **Fast Response**: <5s latency with 7B models
- **Persistent Storage**: All data stored locally in ChromaDB + SQLite

## 🛠️ Tech Stack

| Component      | Technology                  |
| -------------- | --------------------------- |
| **LLM**        | Qwen2.5 (7B/14B) via Ollama |
| **Embeddings** | nomic-embed-text via Ollama |
| **Vector DB**  | ChromaDB (persistent)       |
| **Backend**    | Python + FastAPI            |
| **Frontend**   | Streamlit                   |
| **Metadata**   | SQLite                      |

## 📋 Prerequisites

### 1. Install Ollama

**macOS:**

```bash
# Using Homebrew (recommended)
brew install ollama

# OR download from https://ollama.com/download
# Visit https://ollama.com/download for installation instructions

# Required embedding model
ollama pull nomic-embed-text

# Default LLM model (choose one or more)
ollama pull qwen2.5        # Default (7B)
ollama pull llama3.1       # Alternative (8B)
ollama pull mistral        # Alternative (7B)

# Ensure Python 3.10+ is installed
python3 --version

git clone https://github.com/Hakeem-Sardar/ollama-Qwen2.5-chatbot-local-setup
cd qwen-chatbot

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate

Install Dependencies

pip install -r requirements.txt


qwen-chatbot/
├── data/                 # Source documents (pdf, txt, csv, md)
├── db/                   # ChromaDB vector storage
├── logs/                 # SQLite QA history
├── ingest.py             # Document chunking + embedding
── retriever.py          # Vector search logic
├── chat.py               # Prompt building + LLM calls
├── memory.py             # Conversation history management
├── app.py                # Streamlit UI
├── api.py                # FastAPI backend
├── config.py             # Configuration constants
── requirements.txt      # Python dependencies
└── README.md


Edit config.py to customize settings:

# Model Configuration
LLM_MODEL = "qwen2.5"
EMBEDDING_MODEL = "nomic-embed-text"

# Chunking Configuration
CHUNK_SIZE = 500          # tokens per chunk
CHUNK_OVERLAP = 100       # overlap tokens

# Retrieval Configuration
TOP_K = 4                 # number of chunks to retrieve

# Database Paths
CHROMA_PERSIST_DIR = "db/chroma_db"
SQLITE_DB_PATH = "logs/qa_history.db"

Step 1: Start the Backend (FastAPI)
Open Terminal 1:

# Activate virtual environment
source venv/bin/activate

# Start FastAPI server
uvicorn api:app --reload --port 8000

The API will be available at http://127.0.0.1:8000
API Endpoints:
GET /health - Health check
POST /chat - Send query and get response
GET /history - Retrieve chat history
GET /sessions - List all chat sessions
Step 2: Start the Frontend (Streamlit)


Open Terminal 2 (keep Terminal 1 running):
# Activate virtual environment
source venv/bin/activate

# Start Streamlit UI
streamlit run app.py


The UI will open automatically at http://localhost:8501
Step 3: Upload Documents
Navigate to the 📁 Upload tab in the sidebar
Select PDF, TXT, CSV, or MD files
Click 📤 Upload & Ingest
Wait for the success message showing number of chunks stored

Alternative: Command Line Ingestion
# Add documents to data/ folder
cp your-document.pdf data/

# Run ingestion script
python ingest.py

Start Chatting

Type your question in the chat input
The bot will search your documents and provide a grounded answer
Click ** View Sources** to see which documents were used
Upload more documents anytime to expand knowledge
Step 5: Switch Models (Optional)
Go to ⚙️ Settings tab in sidebar
Select your preferred model from dropdown
Continue chatting with the new model
Available Models:
qwen2.5 (default)
llama3.1
mistral
📱 UI Navigation
The sidebar contains 5 sections:
💬 Chat - Main chat interface with message count
📁 Upload - Upload and ingest new documents
🗂️ Previous Chats - Browse and resume past conversations
📚 Knowledge Base - View all uploaded documents
⚙️ Settings - Switch models, view system info
🔧 API Usage Examples

Using cURL
# Health check
curl http://127.0.0.1:8000/health

# Send a chat query
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the professional summary?",
    "session_id": null,
    "model": "qwen2.5"
  }'

# Get chat history
curl "http://127.0.0.1:8000/history"

# List all sessions
curl "http://127.0.0.1:8000/sessions"

Using Python

import requests

# Send a query
response = requests.post(
    "http://127.0.0.1:8000/chat",
    json={
        "query": "What tools does the candidate use?",
        "model": "qwen2.5"
    }
)

data = response.json()
print(data["reply"])
print(data["sources"])

🔐 Privacy & Security
100% Local: All data stays on your machine
No Cloud: No data sent to external services
Private Storage: Documents, vectors, and chat history stored locally
.gitignore: Sensitive data folders are excluded from Git
Ignored Folders:
data/ - Your documents
db/ - Vector database
logs/ - Chat history
venv/ - Python environment

Troubleshooting
Ollama Not Found
# Check if Ollama is running
ollama list

# Start Ollama server manually
ollama serve

Connection Error

# Ensure FastAPI is running on port 8000
# Check Terminal 1 for errors

# Test API health
curl http://127.0.0.1:8000/health

Model Not Found
# Pull the required model
ollama pull qwen2.5
ollama pull nomic-embed-text

Import Errors

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

ChromaDB Telemetry Warnings
Failed to send telemetry event...
These are harmless warnings from ChromaDB's internal telemetry. They do not affect functionality.

Port Already in Use

# Kill process on port 8000 (macOS/Linux)
lsof -ti:8000 | xargs kill -9

# Or use a different port
uvicorn api:app --reload --port 8001

Performance
Response Time: <5s (with 7B models)
Chunk Size: 500-800 tokens
Embedding Speed: ~100 chunks/minute
Memory Usage: ~4GB RAM (7B model)
🔄 Updating the Database
To add new documents or update existing ones:
Via UI: Use the Upload tab (automatic ingestion)
Via CLI:
   # Add files to data/ folder
   cp new-document.pdf data/

   # Re-run ingestion (safe to re-run)
   python ingest.py?

The system uses upsert with deterministic hashing, so re-running will update without duplicates.
Testing

# Test ingestion
python ingest.py

# Test retrieval
python retriever.py

# Test chat (without UI)
python chat.py

# Test memory
python memory.py

🚀 Future Enhancements
Advanced tokenization (tiktoken)
Multi-file upload with progress bar
Export chat history to JSON/CSV
Advanced search filters
Document deletion from UI
Custom system prompts
Response streaming
Docker containerization
📄 License
MIT License - Feel free to use and modify
Contributing
Fork the repository
Create a feature branch
Commit your changes
Push to the branch
Open a Pull Request
📞 Support
For issues or questions:
Open an issue on GitHub
Check the troubleshooting section
Review the blueprint.md for architecture details
```
