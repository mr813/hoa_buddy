import streamlit as st
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
import json

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def display_document_summary(summary: Dict[str, Any]):
    """Display document processing summary in a nice format."""
    if not summary:
        st.warning("No document summary available")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Chunks", summary.get("total_chunks", 0))
    
    with col2:
        st.metric("Text Length", format_file_size(summary.get("total_text_length", 0)))
    
    with col3:
        st.metric("Avg Chunk Length", f"{summary.get('avg_chunk_length', 0):.0f} chars")
    
    with col4:
        methods = summary.get("processing_methods", [])
        st.metric("Processing Methods", ", ".join(methods) if methods else "Unknown")

def display_search_results(results: List[Dict[str, Any]], message_id: str = ""):
    """Display search results in an expandable format."""
    if not results:
        st.info("No search results found")
        return
    
    for i, result in enumerate(results, 1):
        # Create unique key by combining message_id and result index
        unique_key = f"result_{message_id}_{i}" if message_id else f"result_{i}_{hash(str(result))}"
        
        with st.expander(f"Result {i}: {result['filename']} (Score: {result['score']:.3f})"):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.write("**File:**", result['filename'])
                st.write("**Pages:**", f"{result['page_start']}-{result['page_end']}")
                st.write("**Score:**", f"{result['score']:.3f}")
                st.write("**Method:**", result['processing_method'])
            
            with col2:
                st.text_area(
                    "Content",
                    value=result['text'],
                    height=150,
                    key=unique_key,
                    disabled=True
                )

def display_chat_message(message: Dict[str, Any], is_user: bool = True, message_index: int = 0):
    """Display a chat message with proper formatting."""
    if is_user:
        user_query = message.get('user_query', '')
        st.markdown(f"""
        <div class="user-message">
            <div class="message-header">👤 You</div>
            <div class="message-content">{user_query}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        response_data = message.get('response', {})
        response_text = response_data.get('response', '')
        confidence = response_data.get('confidence', 0.0)
        
        st.markdown(f"""
        <div class="bot-message">
            <div class="message-header">🤖 HOABOT</div>
            <div class="message-content">{response_text}</div>
            <div class="confidence-badge">Confidence: {confidence:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display sources if available
        sources = response_data.get('sources', [])
        if sources:
            with st.expander(f"📚 Sources ({len(sources)} documents)"):
                display_search_results(sources, f"msg_{message_index}")

def create_download_link(data: Dict[str, Any], filename: str) -> str:
    """Create a download link for data."""
    json_str = json.dumps(data, indent=2, default=str)
    b64 = st.b64encode(json_str.encode()).decode()
    return f'<a href="data:file/json;base64,{b64}" download="{filename}">Download {filename}</a>'

def get_current_time() -> str:
    """Get current timestamp as string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def validate_pdf_file(uploaded_file) -> bool:
    """Validate that uploaded file is a PDF."""
    if uploaded_file is None:
        return False
    
    # Check file extension
    if not uploaded_file.name.lower().endswith('.pdf'):
        return False
    
    # Check file size (max 50MB)
    if uploaded_file.size > 50 * 1024 * 1024:
        return False
    
    return True

def create_progress_bar(current: int, total: int, label: str = "Processing"):
    """Create a progress bar with percentage."""
    progress = current / total if total > 0 else 0
    st.progress(progress)
    st.write(f"{label}: {current}/{total} ({progress:.1%})")

def display_error_with_details(error: str, details: str = ""):
    """Display error message with optional details."""
    st.error(f"❌ {error}")
    if details:
        with st.expander("Error Details"):
            st.code(details)

def display_success_message(message: str):
    """Display success message."""
    st.success(f"✅ {message}")

def display_info_message(message: str):
    """Display info message."""
    st.info(f"ℹ️ {message}")

def display_warning_message(message: str):
    """Display warning message."""
    st.warning(f"⚠️ {message}")

def format_timestamp(timestamp: str) -> str:
    """Format timestamp for display."""
    try:
        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%B %d, %Y at %I:%M %p")
    except:
        return timestamp

def create_stats_dataframe(stats: Dict[str, Any]) -> pd.DataFrame:
    """Create a pandas DataFrame from stats for display."""
    if not stats:
        return pd.DataFrame()
    
    data = []
    for key, value in stats.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                data.append({
                    "Metric": f"{key}.{sub_key}",
                    "Value": str(sub_value)
                })
        else:
            data.append({
                "Metric": key,
                "Value": str(value)
            })
    
    return pd.DataFrame(data)
