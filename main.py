import streamlit as st
import sys
import os
from datetime import datetime
from streamlit_option_menu import option_menu

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.config import Config
from app.model_cache import ModelCache
from app.document_processor import DocumentProcessor
from app.vector_store import VectorStore
from app.chat_engine import ChatEngine
from app.utils import *

# Page configuration
st.set_page_config(
    page_title="HOABOT - AI Document Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .chat-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-bottom-right-radius: 5px;
    }
    .bot-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-bottom-left-radius: 5px;
    }
    .message-header {
        font-weight: bold;
        margin-bottom: 8px;
        font-size: 14px;
    }
    .message-content {
        line-height: 1.5;
        font-size: 15px;
    }
    .confidence-badge {
        background: rgba(255,255,255,0.2);
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 12px;
        margin-top: 8px;
        display: inline-block;
    }
    .upload-area {
        border: 2px dashed #ccc;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
        background-color: #fafafa;
    }
    .upload-area:hover {
        border-color: #1f77b4;
        background-color: #f0f8ff;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'model_cache' not in st.session_state:
        st.session_state.model_cache = ModelCache()
    
    if 'document_processor' not in st.session_state:
        st.session_state.document_processor = DocumentProcessor()
    
    if 'vector_store' not in st.session_state:
        try:
            st.session_state.vector_store = VectorStore(st.session_state.model_cache)
        except Exception as e:
            st.error(f"Failed to initialize vector store: {str(e)}")
            st.session_state.vector_store = None
    
    if 'chat_engine' not in st.session_state:
        if st.session_state.vector_store:
            st.session_state.chat_engine = ChatEngine(st.session_state.vector_store)
        else:
            st.session_state.chat_engine = None
    
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'current_time' not in st.session_state:
        st.session_state.current_time = get_current_time()

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 HOABOT</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Document Assistant</p>', unsafe_allow_html=True)
    
    # Validate configuration
    if not Config.validate_config():
        st.error("❌ Configuration validation failed. Please check your API keys and settings.")
        st.stop()
    
    # Model will be loaded when needed (lazy loading)
    st.info("🤖 AI model will be loaded automatically when processing documents.")
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## Navigation")
        selected = option_menu(
            menu_title=None,
            options=["📚 Upload Documents", "💬 Chat", "📊 Analytics", "⚙️ Settings"],
            icons=["upload", "chat", "graph-up", "gear"],
            menu_icon="cast",
            default_index=0,
        )
        
        # Model status indicator
        st.markdown("---")
        st.markdown("### 🤖 Model Status")
        model_info = st.session_state.model_cache.get_model_info()
        if model_info["type"] == "sentence_transformer":
            st.success("✅ Advanced AI Model")
            st.caption(f"Model: {model_info['name']}")
        else:
            st.info("⏳ Model Loading...")
            st.caption("Will load when needed")
        
        st.caption(f"Dimensions: {model_info['dimensions']}")
    
    # Main content based on selection
    if selected == "📚 Upload Documents":
        upload_documents_page()
    elif selected == "💬 Chat":
        chat_page()
    elif selected == "📊 Analytics":
        analytics_page()
    elif selected == "⚙️ Settings":
        settings_page()

def upload_documents_page():
    """Document upload and processing page."""
    st.markdown("## 📚 Document Upload & Processing")
    
    # File upload section
    st.markdown("### Upload PDF Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF documents to process and index"
    )
    
    if uploaded_files:
        st.markdown("### Processing Options")
        col1, col2 = st.columns(2)
        
        with col1:
            chunk_size = st.slider("Chunk Size (tokens)", 200, 500, 384, help="Size of text chunks for processing")
            top_k = st.slider("Search Results (top-k)", 3, 10, 5, help="Number of similar documents to retrieve")
        
        with col2:
            enable_ocr = st.checkbox("Enable OCR Fallback", value=True, help="Use OCR for low-density PDFs")
            auto_process = st.checkbox("Auto-process on upload", value=False, help="Automatically process uploaded files")
        
        # Process files
        if st.button("🚀 Process Documents", type="primary") or auto_process:
            process_uploaded_files(uploaded_files, chunk_size, top_k, enable_ocr)
    
    # Display uploaded files
    if st.session_state.uploaded_files:
        st.markdown("### 📋 Uploaded Documents")
        for file_info in st.session_state.uploaded_files:
            with st.expander(f"📄 {file_info['filename']} - {file_info['status']}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Size:** {format_file_size(file_info['size'])}")
                with col2:
                    st.write(f"**Chunks:** {file_info.get('chunks', 0)}")
                with col3:
                    st.write(f"**Processing:** {file_info.get('method', 'Unknown')}")
                
                if file_info.get('summary'):
                    st.markdown("**Processing Summary:**")
                    display_document_summary(file_info['summary'])

def process_uploaded_files(uploaded_files, chunk_size, top_k, enable_ocr):
    """Process uploaded PDF files."""
    total_files = len(uploaded_files)
    
    for i, uploaded_file in enumerate(uploaded_files):
        st.markdown(f"### Processing {uploaded_file.name} ({i+1}/{total_files})")
        
        # Validate file
        if not validate_pdf_file(uploaded_file):
            st.error(f"❌ Invalid file: {uploaded_file.name}")
            continue
        
        # Process document
        try:
            chunks = st.session_state.document_processor.process_document(uploaded_file)
            
            if chunks:
                # Store in vector database
                if st.session_state.vector_store and st.session_state.vector_store.store_chunks(chunks):
                    # Update session state
                    try:
                        file_info = {
                            'filename': uploaded_file.name,
                            'size': uploaded_file.size,
                            'status': '✅ Processed',
                            'chunks': len(chunks),
                            'method': chunks[0]['metadata'].get('processing_method', 'Unknown') if chunks else 'Unknown',
                            'summary': st.session_state.document_processor.get_document_summary(chunks)
                        }
                        
                        # Check if file already exists
                        existing_files = [f for f in st.session_state.uploaded_files if f['filename'] == uploaded_file.name]
                        if existing_files:
                            # Update existing file
                            for existing_file in st.session_state.uploaded_files:
                                if existing_file['filename'] == uploaded_file.name:
                                    existing_file.update(file_info)
                                    break
                        else:
                            # Add new file
                            st.session_state.uploaded_files.append(file_info)
                        
                        st.success(f"✅ Successfully processed {uploaded_file.name}")
                    except Exception as summary_error:
                        st.warning(f"⚠️ Processed {uploaded_file.name} but couldn't create summary: {str(summary_error)}")
                        # Still add basic file info
                        basic_file_info = {
                            'filename': uploaded_file.name,
                            'size': uploaded_file.size,
                            'status': '✅ Processed (Basic)',
                            'chunks': len(chunks),
                            'method': 'Unknown'
                        }
                        st.session_state.uploaded_files.append(basic_file_info)
                else:
                    st.error(f"❌ Failed to store vectors for {uploaded_file.name}")
            else:
                st.error(f"❌ No chunks generated for {uploaded_file.name}")
                
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            st.info("💡 This might be due to file corruption or unsupported PDF format.")

def chat_page():
    """Chat interface page."""
    st.markdown("## 💬 Chat with HOABOT")
    
    if not st.session_state.chat_engine:
        st.error("❌ Chat engine not available. Please check your configuration.")
        return
    
    # Chat interface with better styling
    st.markdown("### 🤔 Ask Questions About Your Documents")
    
    # Create a container for the chat interface
    with st.container():
        # User input section
        st.markdown("#### 📝 Your Question")
        user_query = st.text_area(
            "Enter your question:",
            placeholder="Ask me anything about your uploaded documents...",
            height=120,
            help="Type your question here and HOABOT will search through your documents to find relevant answers."
        )
        
        # Controls section
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            top_k = st.slider("🔍 Search Results", 3, 10, 5, help="Number of similar documents to search")
        
        with col2:
            use_reranking = st.checkbox("🔄 Reranking", value=True, help="Enable intelligent reranking of search results for better relevance")
        
        with col3:
            if st.button("🤖 Ask HOABOT", type="primary", use_container_width=True):
                if user_query.strip():
                    generate_chat_response(user_query, top_k, use_reranking)
                else:
                    st.warning("Please enter a question.")
        
        with col4:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.conversation_history = []
                st.session_state.chat_engine.clear_conversation_history()
                st.rerun()
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.markdown("---")
        st.markdown("### 💭 Conversation History")
        
        # Create a container for chat messages
        chat_container = st.container()
        
        with chat_container:
            for i, message in enumerate(st.session_state.conversation_history):
                # User message
                display_chat_message(message, is_user=True, message_index=i)
                
                # Bot response
                if 'response' in message:
                    display_chat_message(message, is_user=False, message_index=i)
                
                # Add spacing between message pairs
                if i < len(st.session_state.conversation_history) - 1:
                    st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.info("💡 Start a conversation by asking a question about your uploaded documents!")

def generate_chat_response(user_query, top_k, use_reranking=True):
    """Generate a chat response using the RAG system."""
    try:
        # Update current time
        st.session_state.current_time = get_current_time()
        
        # Generate response
        response_data = st.session_state.chat_engine.generate_response(user_query, top_k, use_reranking)
        
        if 'error' in response_data:
            st.error(f"❌ {response_data['error']}")
        else:
            # Add to conversation history
            conversation_entry = {
                'user_query': user_query,
                'response': response_data,
                'timestamp': st.session_state.current_time
            }
            st.session_state.conversation_history.append(conversation_entry)
            
            # Rerun to display the new message
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error generating response: {str(e)}")

def analytics_page():
    """Analytics and statistics page."""
    st.markdown("## 📊 Analytics & Statistics")
    
    # Vector store statistics
    if st.session_state.vector_store:
        st.markdown("### 🗄️ Vector Database Statistics")
        
        stats = st.session_state.vector_store.get_index_stats()
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Vectors", stats.get("total_vector_count", 0))
            
            with col2:
                st.metric("Dimensions", stats.get("dimension", 384))
            
            with col3:
                fullness = stats.get("index_fullness", 0)
                st.metric("Index Fullness", f"{fullness:.1%}")
            
            with col4:
                namespaces = stats.get("namespaces", {})
                st.metric("Namespaces", len(namespaces))
            
            # Detailed stats table
            if stats:
                st.markdown("### 📋 Detailed Statistics")
                df = create_stats_dataframe(stats)
                st.dataframe(df, use_container_width=True)
        else:
            st.warning("No statistics available")
    
    # Document statistics
    if st.session_state.uploaded_files:
        st.markdown("### 📚 Document Statistics")
        
        total_files = len(st.session_state.uploaded_files)
        total_chunks = sum(f.get('chunks', 0) for f in st.session_state.uploaded_files)
        total_size = sum(f.get('size', 0) for f in st.session_state.uploaded_files)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Files", total_files)
        
        with col2:
            st.metric("Total Chunks", total_chunks)
        
        with col3:
            st.metric("Total Size", format_file_size(total_size))
        
        # Files table
        st.markdown("### 📋 Document Details")
        files_data = []
        for file_info in st.session_state.uploaded_files:
            files_data.append({
                "Filename": file_info['filename'],
                "Size": format_file_size(file_info['size']),
                "Status": file_info['status'],
                "Chunks": file_info.get('chunks', 0),
                "Method": file_info.get('method', 'Unknown')
            })
        
        if files_data:
            df = pd.DataFrame(files_data)
            st.dataframe(df, use_container_width=True)
    
    # Model information
    st.markdown("### 🤖 Model Information")
    model_info = st.session_state.model_cache.get_model_info()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Model Type:** {model_info.get('type', 'Unknown')}")
        st.write(f"**Dimensions:** {model_info.get('dimensions', 384)}")
    
    with col2:
        if model_info.get('name'):
            st.write(f"**Model Name:** {model_info['name']}")
        st.write(f"**Cache Directory:** {st.session_state.model_cache.cache_dir}")

def settings_page():
    """Settings and configuration page."""
    st.markdown("## ⚙️ Settings & Configuration")
    
    # Configuration status
    st.markdown("### 🔧 Configuration Status")
    
    pinecone_config = Config.get_pinecone_config()
    perplexity_config = Config.get_perplexity_config()
    email_config = Config.get_email_config()
    
    # Pinecone configuration
    with st.expander("🗄️ Pinecone Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**API Key:** {'✅ Configured' if pinecone_config['api_key'] else '❌ Missing'}")
            st.write(f"**Index Name:** {pinecone_config['index_name']}")
        
        with col2:
            st.write(f"**Environment:** {pinecone_config['environment']}")
            st.write(f"**Dimensions:** {pinecone_config['dimension']}")
    
    # Perplexity configuration
    with st.expander("🤖 Perplexity Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**API Key:** {'✅ Configured' if perplexity_config['api_key'] else '❌ Missing'}")
            st.write(f"**Model:** {perplexity_config['model']}")
        
        with col2:
            st.write(f"**Base URL:** {perplexity_config['base_url']}")
            st.write(f"**Timeout:** {perplexity_config['timeout']}s")
    
    # Email configuration (optional)
    with st.expander("📧 Email Configuration (Optional)", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**SMTP Server:** {email_config['smtp_server']}")
            st.write(f"**SMTP Port:** {email_config['smtp_port']}")
        
        with col2:
            st.write(f"**Username:** {'✅ Configured' if email_config['smtp_username'] else '❌ Missing'}")
            st.write(f"**From Email:** {email_config['from_email']}")
    
    # System Prompt Configuration
    st.markdown("### 🤖 AI Assistant Configuration")
    
    with st.expander("💬 System Prompt Settings", expanded=True):
        st.markdown("""
        **Customize how HOABOT responds to your questions.** 
        
        The system prompt defines the AI assistant's personality, behavior, and response style. 
        You can modify it to make HOABOT more formal, casual, detailed, or focused on specific topics.
        """)
        
        # Get current system prompt
        current_prompt = Config.get_system_prompt()
        
        # System prompt editor
        new_prompt = st.text_area(
            "System Prompt",
            value=current_prompt,
            height=300,
            help="Define how HOABOT should behave and respond to questions. This affects all future conversations.",
            placeholder="Enter your custom system prompt here..."
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Changes", type="primary"):
                if new_prompt.strip():
                    Config.set_system_prompt(new_prompt.strip())
                    st.success("✅ System prompt updated successfully!")
                    st.rerun()
                else:
                    st.error("❌ System prompt cannot be empty!")
        
        with col2:
            if st.button("🔄 Reset to Default", type="secondary"):
                Config.set_system_prompt("")  # This will trigger the default prompt
                st.success("✅ System prompt reset to default!")
                st.rerun()
        
        with col3:
            if st.button("📋 Copy Current", type="secondary"):
                st.write("Current system prompt copied to clipboard!")
                st.code(current_prompt)
        
        # Add a fourth column for saving as default
        col4 = st.columns(1)[0]
        with col4:
            if st.button("💾 Save as Default", type="primary"):
                Config.save_current_as_default()
                st.rerun()
        
        # Show prompt statistics
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Prompt Length", f"{len(current_prompt)} characters")
        
        with col2:
            word_count = len(current_prompt.split())
            st.metric("Word Count", f"{word_count} words")
        
        with col3:
            line_count = len(current_prompt.split('\n'))
            st.metric("Line Count", f"{line_count} lines")
    
    # Search Configuration
    st.markdown("### 🔍 Search & Reranking Configuration")
    
    with st.expander("🔄 Reranking Settings", expanded=True):
        st.markdown("""
        **Configure how HOABOT searches and ranks document results.**
        
        Reranking improves search quality by combining semantic similarity with keyword matching, 
        content quality assessment, and document importance scoring.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            default_reranking = st.checkbox(
                "Enable Reranking by Default", 
                value=True, 
                help="Automatically enable reranking for all searches"
            )
            
            if st.button("💾 Save Reranking Settings", type="primary"):
                st.session_state["default_reranking"] = default_reranking
                st.success("✅ Reranking settings saved!")
        
        with col2:
            st.markdown("**Reranking Components:**")
            st.markdown("""
            - **Semantic Similarity (50%)**: Vector-based similarity
            - **Keyword Relevance (30%)**: Exact phrase and word matching
            - **Content Quality (15%)**: Text length, diversity, processing method
            - **Document Importance (5%)**: Bylaws preference, page position
            """)
    
    # Database management
    st.markdown("### 🗄️ Database Management")
    
    if st.session_state.vector_store:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear All Vectors", type="secondary"):
                if st.session_state.vector_store.clear_all_vectors():
                    st.session_state.uploaded_files = []
                    st.rerun()
        
        with col2:
            if st.button("📊 Refresh Statistics", type="secondary"):
                st.rerun()
    
    # Export/Import
    st.markdown("### 📤 Export/Import")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.conversation_history:
            conversation_data = {
                'conversation_history': st.session_state.conversation_history,
                'exported_at': get_current_time()
            }
            
            st.download_button(
                label="📥 Export Chat History",
                data=json.dumps(conversation_data, indent=2, default=str),
                file_name=f"hoabot_chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.session_state.uploaded_files:
            files_data = {
                'uploaded_files': st.session_state.uploaded_files,
                'exported_at': get_current_time()
            }
            
            st.download_button(
                label="📥 Export File List",
                data=json.dumps(files_data, indent=2, default=str),
                file_name=f"hoabot_files_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
