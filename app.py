import streamlit as st
import requests

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000/chat"

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Qwen RAG Chatbot", page_icon="🤖", layout="centered")

# --- Initialize Session State ---
# We use session state to hold the UI chat history and the backend session_id
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# --- UI Header ---
st.title("🤖 Qwen RAG Chatbot")
st.caption("Grounded in your documents. Powered by Qwen2.5, ChromaDB, and Ollama.")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Controls")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        # Reset both UI history and backend session ID
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If it's an assistant message and has sources, show them in an expander
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📄 View Sources"):
                for src in message["sources"]:
                    st.markdown(f"- {src}")

# --- Chat Input & API Integration ---
if prompt := st.chat_input("Ask a question about your documents..."):
    # 1. Add and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating response..."):
            try:
                # Prepare the payload, including the session_id if we have one
                payload = {
                    "query": prompt,
                    "session_id": st.session_state.session_id
                }
                
                # Call the FastAPI backend
                response = requests.post(API_URL, json=payload)
                response.raise_for_status() # Raise error for bad status codes (4xx, 5xx)
                data = response.json()

                reply = data["reply"]
                sources = data["sources"]
                
                # CRITICAL: Capture the session_id from the first response for future turns
                if not st.session_state.session_id:
                    st.session_state.session_id = data["session_id"]

                # Display the reply
                st.markdown(reply)
                
                # Display sources if they exist
                if sources:
                    with st.expander("📄 View Sources"):
                        for src in sources:
                            st.markdown(f"- {src}")
                            
                # 3. Save assistant message to UI history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": reply, 
                    "sources": sources
                })

            except requests.exceptions.ConnectionError:
                st.error("❌ **Connection Error:** Could not connect to the FastAPI backend. Is `uvicorn api:app` running on port 8000?")
            except Exception as e:
                st.error(f"❌ **Error:** An unexpected error occurred: {str(e)}")