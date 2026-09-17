import os
import streamlit as st
import requests
from config import DATA_DIR
from ingest import ingest_single_file

API_URL = "http://127.0.0.1:8000/chat"
HISTORY_URL = "http://127.0.0.1:8000/history"
SESSIONS_URL = "http://127.0.0.1:8000/sessions"

st.set_page_config(page_title="Qwen RAG Chatbot", page_icon="🤖", layout="centered")

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "history_loaded" not in st.session_state:
    st.session_state.history_loaded = False
if "sidebar_expanded" not in st.session_state:
    st.session_state.sidebar_expanded = "chat"  # Default to chat section

# --- RESTORE HISTORY ON PAGE LOAD ---
if not st.session_state.history_loaded:
    try:
        res = requests.get(HISTORY_URL)
        if res.status_code == 200:
            data = res.json()
            if data["messages"]:
                st.session_state.session_id = data["session_id"]
                st.session_state.messages = data["messages"]
    except Exception:
        pass
    st.session_state.history_loaded = True

# --- UI Header ---
st.title("🤖 Qwen RAG Chatbot")
st.caption("Grounded in your documents. Powered by Qwen2.5, ChromaDB, and Ollama.")

# --- REDESIGNED SIDEBAR ---
with st.sidebar:
    st.title("📋 Menu")
    
    # Navigation Radio Buttons
    nav_option = st.radio(
        "Navigate",
        ["💬 Chat", "📁 Upload", "🗂️ Previous Chats", "📚 Knowledge Base", "⚙️ Settings"],
        label_visibility="collapsed",
        index=0
    )
    
    st.divider()
    
    # === CHAT SECTION (Default) ===
    if nav_option == " Chat":
        st.subheader(" Chat Interface")
        st.info("Ask questions about your documents below.")
        
        # Quick stats
        if st.session_state.messages:
            st.metric("Messages in this chat", len(st.session_state.messages))
        
        if st.button("🗑️ Clear Current Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = None
            st.rerun()
    
    # === UPLOAD SECTION ===
    elif nav_option == "📁 Upload":
        st.subheader("📁 Upload Documents")
        st.markdown("Add new documents to the knowledge base.")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "txt", "csv", "md"],
            help="Supported formats: PDF, TXT, CSV, MD",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            save_path = DATA_DIR / uploaded_file.name
            
            if st.button("📤 Upload & Ingest", use_container_width=True, type="primary"):
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        with open(save_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        num_chunks = ingest_single_file(str(save_path))
                        
                        if num_chunks > 0:
                            st.success(f"✅ Ingested {num_chunks} chunks!")
                        else:
                            st.warning("️ No text found in file.")
                        
                        time.sleep(1)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        st.markdown("---")
        st.caption("Files are saved to `data/` folder")
    
    # === PREVIOUS CHATS SECTION ===
    elif nav_option == "️ Previous Chats":
        st.subheader("🗂️ Chat History")
        
        try:
            sessions_resp = requests.get(SESSIONS_URL, timeout=2)
            if sessions_resp.status_code == 200:
                sessions = sessions_resp.json()
                
                if sessions:
                    # New Chat button at top
                    if st.button("➕ New Chat", use_container_width=True, type="primary"):
                        st.session_state.messages = []
                        st.session_state.session_id = None
                        st.rerun()
                    
                    st.markdown("---")
                    st.caption(f"{len(sessions)} previous conversations")
                    
                    # Display sessions as clickable cards
                    for i, sess in enumerate(sessions[:20], 1):  # Limit to 20 most recent
                        preview = sess['first_query'][:40] + "..." if len(sess['first_query']) > 40 else sess['first_query']
                        
                        # Create a container for each session
                        with st.container():
                            is_current = sess['session_id'] == st.session_state.session_id
                            
                            if is_current:
                                st.markdown(f"**▶️ {preview}**")
                                st.markdown(f"<small>{sess['message_count']} messages</small>", unsafe_allow_html=True)
                            else:
                                if st.button(f"💬 {preview}", key=f"sess_{sess['session_id']}", use_container_width=True):
                                    try:
                                        hist_resp = requests.get(
                                            f"{HISTORY_URL}?session_id={sess['session_id']}",
                                            timeout=5
                                        )
                                        if hist_resp.status_code == 200:
                                            data = hist_resp.json()
                                            st.session_state.session_id = data["session_id"]
                                            st.session_state.messages = data["messages"]
                                            st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to load: {e}")
                            
                            st.markdown(f"<small style='color: gray'>{sess['message_count']} msgs</small>", unsafe_allow_html=True)
                            st.divider()
                else:
                    st.info("No previous chats yet.")
                    
        except Exception as e:
            st.error("Couldn't load chat history")
    
    # === KNOWLEDGE BASE SECTION ===
    elif nav_option == "📚 Knowledge Base":
        st.subheader("📚 Documents")
        
        data_files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(DATA_DIR / f)]
        
        if data_files:
            st.markdown(f"**{len(data_files)} documents** loaded")
            st.divider()
            
            for f in data_files:
                file_size = os.path.getsize(DATA_DIR / f) / 1024  # KB
                with st.container():
                    st.markdown(f"📄 **{f}**")
                    st.caption(f"{file_size:.1f} KB")
                    st.divider()
        else:
            st.warning("No documents uploaded yet.")
            st.info("Go to **Upload** tab to add documents.")
    
    # === SETTINGS SECTION ===
    elif nav_option == "⚙️ Settings":
        st.subheader("️ Settings")
        
        st.markdown("**System Info**")
        st.json({
            "Backend": "FastAPI",
            "LLM": "Qwen2.5",
            "Vector DB": "ChromaDB",
            "Embeddings": "nomic-embed-text"
        })
        
        st.divider()
        
        if st.button("🔄 Reset Everything", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = None
            st.session_state.history_loaded = False
            st.rerun()

# === MAIN CHAT INTERFACE ===
# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📄 View Sources", expanded=False):
                for src in message["sources"]:
                    st.markdown(f"- {src}")

# Chat Input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating..."):
            try:
                payload = {
                    "query": prompt,
                    "session_id": st.session_state.session_id
                }

                response = requests.post(API_URL, json=payload)
                response.raise_for_status()
                data = response.json()

                reply = data["reply"]
                sources = data["sources"]

                if not st.session_state.session_id:
                    st.session_state.session_id = data["session_id"]

                st.markdown(reply)

                if sources:
                    with st.expander("📄 View Sources", expanded=False):
                        for src in sources:
                            st.markdown(f"- {src}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply,
                    "sources": sources
                })

            except requests.exceptions.ConnectionError:
                st.error("❌ **Connection Error:** Is the FastAPI backend running on port 8000?")
            except Exception as e:
                st.error(f"❌ **Error:** {str(e)}")