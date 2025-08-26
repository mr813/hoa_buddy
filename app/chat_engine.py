import requests
import streamlit as st
import json
from typing import List, Dict, Any, Optional
from .config import Config
from .vector_store import VectorStore

class ChatEngine:
    """Handles RAG chat functionality with Perplexity API integration."""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.perplexity_config = Config.get_perplexity_config()
        self.conversation_history = []
    
    def generate_response(self, user_query: str, top_k: int = 5, use_reranking: bool = True) -> Dict[str, Any]:
        """Generate a response using RAG with Perplexity API."""
        if not user_query.strip():
            return {"error": "Empty query provided"}
        
        try:
            # Step 1: Search for relevant documents with reranking
            with st.spinner("Searching for relevant documents..."):
                search_results = self.vector_store.search_similar(user_query, top_k=top_k, use_reranking=use_reranking)
            
            if not search_results:
                return {
                    "response": "I couldn't find any relevant documents to answer your question. Please try uploading some documents first or rephrase your question.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Step 2: Prepare context from search results
            context = self._prepare_context(search_results)
            
            # Step 3: Generate enhanced prompt
            enhanced_prompt = self._create_enhanced_prompt(user_query, context)
            
            # Step 4: Call Perplexity API
            with st.spinner("Generating response..."):
                api_response = self._call_perplexity_api(enhanced_prompt)
            
            # Step 5: Process and return response
            response_data = {
                "response": api_response.get("response", "Sorry, I couldn't generate a response."),
                "sources": search_results,
                "confidence": self._calculate_confidence(search_results),
                "context_used": context[:500] + "..." if len(context) > 500 else context
            }
            
            # Add to conversation history
            self.conversation_history.append({
                "user_query": user_query,
                "response": response_data,
                "timestamp": st.session_state.get("current_time", "Unknown")
            })
            
            return response_data
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return {"error": f"Failed to generate response: {str(e)}"}
    
    def _prepare_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Prepare context from search results."""
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            context_part = f"Document {i} (Score: {result['score']:.3f}, File: {result['filename']}, Pages: {result['page_start']}-{result['page_end']}):\n{result['text']}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _create_enhanced_prompt(self, user_query: str, context: str) -> str:
        """Create an enhanced prompt combining user query with retrieved context."""
        system_prompt = Config.get_system_prompt()
        
        prompt = f"""{system_prompt}

Context from documents:
{context}

User Question: {user_query}

Answer:"""
        
        return prompt
    
    def _call_perplexity_api(self, prompt: str) -> Dict[str, Any]:
        """Call the Perplexity API to generate a response."""
        headers = {
            "Authorization": f"Bearer {self.perplexity_config['api_key']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.perplexity_config["model"],
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(
                self.perplexity_config["base_url"],
                headers=headers,
                json=data,
                timeout=self.perplexity_config["timeout"]
            )
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return {
                        "response": result["choices"][0]["message"]["content"],
                        "usage": result.get("usage", {}),
                        "model": result.get("model", "")
                    }
                else:
                    raise Exception("No response content in API response")
            else:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            raise Exception("API request timed out")
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}")
    
    def _calculate_confidence(self, search_results: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on search results."""
        if not search_results:
            return 0.0
        
        # Calculate average similarity score
        scores = [result["score"] for result in search_results]
        avg_score = sum(scores) / len(scores)
        
        # Normalize to 0-1 range (assuming cosine similarity scores)
        confidence = max(0.0, min(1.0, avg_score))
        
        return confidence
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history."""
        return self.conversation_history
    
    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
    
    def get_response_summary(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the response data."""
        return {
            "response_length": len(response_data.get("response", "")),
            "num_sources": len(response_data.get("sources", [])),
            "confidence": response_data.get("confidence", 0.0),
            "has_error": "error" in response_data,
            "top_source_score": max([s["score"] for s in response_data.get("sources", [])], default=0.0)
        }
