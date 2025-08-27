"""
🔍 Augmented Research Assistant - Streamlit RAG App
A modern, dark-themed RAG application with LangChain + LlamaIndex + Gemini API
"""

import streamlit as st
import os
from pathlib import Path
import time
from typing import List

# Import our services
from config.settings import config, validate_config, get_supported_file_types
from services.document_processor import DocumentProcessor
from services.vector_store import VectorStoreService
from core.rag_chain import RAGChain
from utils.cache_manager import CacheManager

# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and modern styling
def load_custom_css():
    st.markdown("""
    <style>
    /* Dark theme customization */
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e2329 0%, #2d3748 100%);
    }
    
    /* Chat messages */
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #4f46e5;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
    }
    
    .user-message {
        border-left-color: #10b981;
        background: rgba(16, 185, 129, 0.1);
    }
    
    .assistant-message {
        border-left-color: #6366f1;
        background: rgba(99, 102, 241, 0.1);
    }
    
    /* Source cards */
    .source-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Upload area */
    .upload-area {
        border: 2px dashed #4f46e5;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: rgba(79, 70, 229, 0.1);
        transition: all 0.3s ease;
    }
    
    /* Metrics cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Custom buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* File uploader */
    .stFileUploader > div > div {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        border: 2px dashed #4f46e5;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    </style>
    """, unsafe_allow_html=True)

def get_persistent_session_id():
    """Get or create a persistent session ID based on machine characteristics."""
    import platform
    import hashlib
    import os
    
    # Create a persistent ID based on machine characteristics and user
    machine_info = f"{platform.node()}_{platform.system()}_{os.getlogin() if hasattr(os, 'getlogin') else 'user'}"
    machine_hash = hashlib.md5(machine_info.encode()).hexdigest()[:8]
    persistent_id = f"ara_{machine_hash}"
    
    return persistent_id

# Initialize session state with persistent session ID
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = 0
    if "vector_store_ready" not in st.session_state:
        st.session_state.vector_store_ready = False
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "user_session_id" not in st.session_state:
        st.session_state.user_session_id = get_persistent_session_id()

def check_existing_documents(vector_store):
    """Check if documents already exist in vector store and update session state."""
    try:
        collection_info = vector_store.get_collection_info()
        document_count = collection_info.get("document_count", 0)
        
        if document_count > 0:
            st.session_state.vector_store_ready = True
            if st.session_state.documents_processed == 0:
                st.session_state.documents_processed = document_count
        
        return document_count > 0
    except Exception:
        return False

# Initialize services
@st.cache_resource
def initialize_services():
    """Initialize all services with caching - using persistent session ID."""
    # Use a consistent session ID for all tabs and browser instances
    persistent_session_id = get_persistent_session_id()
    
    doc_processor = DocumentProcessor()
    vector_store = VectorStoreService(user_session_id=persistent_session_id)
    cache_manager = CacheManager(user_session_id=persistent_session_id)
    rag_chain = RAGChain(vector_store, cache_manager)
    
    return doc_processor, vector_store, rag_chain, cache_manager, persistent_session_id

def display_header():
    """Display the app header with animations."""
    st.markdown("""
    <div class="fade-in">
        <h1 style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                   font-size: 3rem; margin-bottom: 0;">
            🔍 Augmented Research Assistant
        </h1>
        <p style="text-align: center; color: #a0aec0; font-size: 1.2rem; margin-top: 0;">
            Upload documents and chat with your knowledge base using AI
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar(vector_store, cache_manager):
    """Display the sidebar with file upload and system info."""
    with st.sidebar:
        st.markdown("### 📁 Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=get_supported_file_types(),
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, DOCX, MD"
        )
        
        if uploaded_files:
            process_uploaded_files(uploaded_files)
        
        # Collection info
        st.markdown("### 📊 Knowledge Base Status")
        collection_info = vector_store.get_collection_info()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", collection_info["document_count"])
        with col2:
            st.metric("Files Processed", len(st.session_state.uploaded_files))
        
        # Display user session info
        st.markdown("### 👤 Session Info")
        st.text(f"Session ID: {st.session_state.user_session_id}")
        st.caption("Your documents are private to this session")
        
        # Cache info
        st.markdown("### ⚡ Cache Statistics")
        cache_stats = cache_manager.get_cache_stats()
        st.metric("Cache Size", f"{cache_stats['total_size_mb']:.2f} MB")
        
        # Management buttons
        st.markdown("### 🛠️ Management")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear Documents", type="secondary"):
                if vector_store.clear_collection():
                    st.session_state.documents_processed = 0
                    st.session_state.uploaded_files = []
                    st.session_state.vector_store_ready = False
                    st.success("Documents cleared!")
                    st.rerun()
        
        with col2:
            if st.button("Clear Cache", type="secondary"):
                if cache_manager.clear_cache():
                    st.success("Cache cleared!")

def process_uploaded_files(uploaded_files):
    """Process uploaded files and add to vector store."""
    doc_processor, vector_store, _, _, _ = initialize_services()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    new_files = []
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.uploaded_files:
            new_files.append(uploaded_file)
    
    if not new_files:
        st.info("All files already processed!")
        return
    
    for i, uploaded_file in enumerate(new_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        
        # Process document
        documents = doc_processor.process_uploaded_file(uploaded_file)
        
        if documents:
            # Add to vector store
            if vector_store.add_documents(documents):
                st.session_state.uploaded_files.append(uploaded_file.name)
                st.session_state.documents_processed += len(documents)
                st.session_state.vector_store_ready = True
        
        progress_bar.progress((i + 1) / len(new_files))
    
    status_text.text("✅ Processing complete!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()
    
    st.success(f"Processed {len(new_files)} new files!")
    st.rerun()

def display_chat_messages():
    """Display chat messages only."""
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources", expanded=False):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>Source {i}: {source['source_file']}</strong><br>
                            <small>Relevance: {source['relevance_score']:.3f}</small><br>
                            <p style="margin-top: 0.5rem;">{source['content']}</p>
                        </div>
                        """, unsafe_allow_html=True)

def handle_chat_input(rag_chain):
    """Handle chat input and response generation."""
    # Chat input at bottom
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.vector_store_ready:
            st.warning("Please upload some documents first!")
            return
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag_chain.query(prompt)
            
            st.markdown(response["answer"])
            
            # Add assistant message with sources
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response["answer"],
                "sources": response["sources"]
            })
            
            # Display sources
            if response["sources"]:
                with st.expander("📚 Sources", expanded=False):
                    for i, source in enumerate(response["sources"], 1):
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>Source {i}: {source['source_file']}</strong><br>
                            <small>Relevance: {source['relevance_score']:.3f}</small><br>
                            <p style="margin-top: 0.5rem;">{source['content']}</p>
                        </div>
                        """, unsafe_allow_html=True)

def display_api_key_setup():
    """Display API key setup interface."""
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h2>🔑 Setup Required</h2>
        <p>Please set your Google API key to get started.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### How to get your Google API Key:")
    st.markdown("""
    1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
    2. Click "Create API Key"
    3. Copy your API key
    4. Set it as an environment variable: `GOOGLE_API_KEY=your_key_here`
    """)
    
    # Allow manual input for testing
    api_key = st.text_input("Or enter your API key here:", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        config.GOOGLE_API_KEY = api_key
        st.success("API key set! Please refresh the page.")
        st.rerun()

def main():
    """Main application function."""
    load_custom_css()
    initialize_session_state()
    
    # Check API key
    if not validate_config():
        display_api_key_setup()
        return
    
    # Initialize services
    doc_processor, vector_store, rag_chain, cache_manager, persistent_session_id = initialize_services()
    
    # Update session state with persistent ID
    st.session_state.user_session_id = persistent_session_id
    
    # Check if services are initialized
    if not vector_store.is_initialized() or not rag_chain.is_initialized():
        st.error("Failed to initialize services. Please check your API key and try again.")
        return
    
    # Check for existing documents in vector store
    check_existing_documents(vector_store)
    
    # Display header
    display_header()
    
    # Main layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 💬 Chat with your Documents")
        
        # Create a container for messages with fixed height
        message_container = st.container()
        with message_container:
            display_chat_messages()
    
    with col2:
        display_sidebar(vector_store, cache_manager)
    
    # Chat input at the bottom (outside columns for full width)
    handle_chat_input(rag_chain)

if __name__ == "__main__":
    main()
