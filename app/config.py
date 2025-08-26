import streamlit as st
import os
from typing import Dict, Any

class Config:
    """Configuration management for the RAG application."""
    
    @staticmethod
    def get_pinecone_config() -> Dict[str, Any]:
        """Get Pinecone configuration from Streamlit secrets."""
        return {
            "api_key": st.secrets.get("pinecone", {}).get("api_key"),
            "index_name": st.secrets.get("pinecone", {}).get("index_name", "hoa-bot"),
            "environment": st.secrets.get("pinecone", {}).get("environment", "us-east1-aws"),
            "dimension": 384,
            "metric": "cosine"
        }
    
    @staticmethod
    def get_perplexity_config() -> Dict[str, Any]:
        """Get Perplexity configuration from Streamlit secrets."""
        return {
            "api_key": st.secrets.get("perplexity", {}).get("api_key"),
            "model": st.secrets.get("perplexity", {}).get("model", "sonar"),
            "base_url": st.secrets.get("perplexity", {}).get("base_url", "https://api.perplexity.ai/chat/completions"),
            "timeout": 10,
            "max_retries": 3
        }
    
    @staticmethod
    def get_email_config() -> Dict[str, Any]:
        """Get email configuration from Streamlit secrets."""
        return {
            "smtp_server": st.secrets.get("email", {}).get("smtp_server", "smtp.gmail.com"),
            "smtp_port": st.secrets.get("email", {}).get("smtp_port", 587),
            "smtp_username": st.secrets.get("email", {}).get("smtp_username"),
            "smtp_password": st.secrets.get("email", {}).get("smtp_password"),
            "from_email": st.secrets.get("email", {}).get("from_email")
        }
    
    @staticmethod
    def get_system_prompt() -> str:
        """Get the current system prompt from session state or return default."""
        return st.session_state.get("system_prompt", Config._default_prompt)
    
    @staticmethod
    def set_system_prompt(prompt: str) -> None:
        """Set the system prompt in session state."""
        st.session_state["system_prompt"] = prompt
    
    @staticmethod
    def save_current_as_default() -> None:
        """Save the current system prompt as the new default."""
        current_prompt = st.session_state.get("system_prompt")
        if current_prompt:
            # Update the default prompt in the function
            Config._default_prompt = current_prompt
            st.success("✅ Current system prompt saved as new default!")
        else:
            st.warning("No current system prompt to save.")
    
    # Class variable to store the default prompt
    _default_prompt = """You are HOABOT, a helpful AI assistant that answers questions based on provided document context. 

Instructions:
1. Answer the user's question based ONLY on the provided context
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Cite specific documents and page numbers when possible
4. Be concise but thorough
5. If you're unsure about something, acknowledge the uncertainty
6. Use a professional but friendly tone
7. Focus on providing accurate, helpful information from the documents
8. Finally compare any discrepancies found between the info retrieved from other documents and the "Condo Docs - Casitas Del Mar" Which are the bylaws. Also flag any rules or regulations that expand the scope of the bylaws. For example, the rental policy requires board approval but the bylaws only require board approval for leases > 1 year. Instance like this I want you to flag them.
9. Also cross reference current Florida Condo HOA laws and flag any potential discrepancies. Please cite specific statutes"""
    
    @staticmethod
    def validate_config() -> bool:
        """Validate that all required configuration is present."""
        pinecone_config = Config.get_pinecone_config()
        perplexity_config = Config.get_perplexity_config()
        
        required_pinecone = ["api_key", "index_name", "environment"]
        required_perplexity = ["api_key"]
        
        for key in required_pinecone:
            if not pinecone_config.get(key):
                st.error(f"Missing required Pinecone configuration: {key}")
                return False
        
        for key in required_perplexity:
            if not perplexity_config.get(key):
                st.error(f"Missing required Perplexity configuration: {key}")
                return False
        
        return True
