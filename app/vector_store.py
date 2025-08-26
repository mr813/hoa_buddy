import pinecone
import streamlit as st
import numpy as np
from typing import List, Dict, Any, Optional
import uuid
from .config import Config
from .model_cache import ModelCache

class VectorStore:
    """Handles Pinecone vector database operations."""
    
    def __init__(self, model_cache: ModelCache):
        self.model_cache = model_cache
        self.pinecone_config = Config.get_pinecone_config()
        self.pinecone_client = None
        self.index = None
        self._initialize_pinecone()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone client and index."""
        try:
            # Initialize Pinecone client with new API
            self.pinecone_client = pinecone.Pinecone(
                api_key=self.pinecone_config["api_key"]
            )
            
            # Check if index exists
            existing_indexes = [index.name for index in self.pinecone_client.list_indexes()]
            if self.pinecone_config["index_name"] not in existing_indexes:
                st.info(f"Creating Pinecone index: {self.pinecone_config['index_name']}")
                self.pinecone_client.create_index(
                    name=self.pinecone_config["index_name"],
                    dimension=self.pinecone_config["dimension"],
                    metric=self.pinecone_config["metric"],
                    spec=pinecone.ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
            
            # Connect to index
            self.index = self.pinecone_client.Index(self.pinecone_config["index_name"])
            st.success("Pinecone connection established!")
            
        except Exception as e:
            st.error(f"Failed to initialize Pinecone: {str(e)}")
            raise
    
    def store_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """Store document chunks in Pinecone with embeddings."""
        if not chunks or not self.index:
            return False
        
        try:
            # Prepare data for batch upsert
            vectors = []
            texts = [chunk["text"] for chunk in chunks]
            
            # Generate embeddings
            with st.spinner("Generating embeddings..."):
                embeddings = self.model_cache.get_embeddings(texts)
            
            # Prepare vectors for Pinecone
            for i, chunk in enumerate(chunks):
                try:
                    vector_id = str(uuid.uuid4())
                    vector_data = {
                        "id": vector_id,
                        "values": embeddings[i].tolist(),
                        "metadata": {
                            "text": chunk["text"],
                            "chunk_id": chunk["metadata"].get("chunk_id", f"chunk_{i}"),
                            "filename": chunk["metadata"].get("filename", "unknown"),
                            "page_start": chunk["metadata"].get("page_start", 1),
                            "page_end": chunk["metadata"].get("page_end", 1),
                            "chunk_index": chunk["metadata"].get("chunk_index", i),
                            "text_length": chunk["metadata"].get("text_length", len(chunk["text"])),
                            "processing_method": chunk["metadata"].get("processing_method", "unknown")
                        }
                    }
                    vectors.append(vector_data)
                except Exception as chunk_error:
                    st.warning(f"Error processing chunk {i}: {str(chunk_error)}")
                    continue
            
            # Upsert to Pinecone with new API
            with st.spinner("Storing vectors in Pinecone..."):
                self.index.upsert(vectors=vectors)
            
            st.success(f"Successfully stored {len(vectors)} vectors in Pinecone!")
            return True
            
        except Exception as e:
            st.error(f"Failed to store chunks in Pinecone: {str(e)}")
            return False
    
    def search_similar(self, query: str, top_k: int = 5, use_reranking: bool = True) -> List[Dict[str, Any]]:
        """Search for similar documents using semantic similarity with optional reranking."""
        if not self.index or not query.strip():
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.model_cache.get_embeddings(query)
            
            # Search in Pinecone with new API - get more results for reranking
            initial_k = top_k * 3 if use_reranking else top_k
            search_results = self.index.query(
                vector=query_embedding[0].tolist(),
                top_k=initial_k,
                include_metadata=True
            )
            
            # Format initial results
            results = []
            for match in search_results.matches:
                result = {
                    "score": match.score,
                    "text": match.metadata.get("text", ""),
                    "filename": match.metadata.get("filename", ""),
                    "page_start": match.metadata.get("page_start", 0),
                    "page_end": match.metadata.get("page_end", 0),
                    "chunk_index": match.metadata.get("chunk_index", 0),
                    "text_length": match.metadata.get("text_length", 0),
                    "processing_method": match.metadata.get("processing_method", "")
                }
                results.append(result)
            
            # Apply reranking if enabled
            if use_reranking and len(results) > top_k:
                results = self._rerank_results(query, results, top_k)
            
            return results[:top_k]
            
        except Exception as e:
            st.error(f"Failed to search in Pinecone: {str(e)}")
            return []
    
    def _rerank_results(self, query: str, results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Rerank search results using a hybrid scoring approach."""
        if not results:
            return results
        
        try:
            # Calculate additional scoring factors
            for result in results:
                # Get the original semantic score
                semantic_score = result["score"]
                
                # Calculate text relevance score (keyword matching)
                text_relevance = self._calculate_text_relevance(query, result["text"])
                
                # Calculate content quality score
                quality_score = self._calculate_content_quality(result)
                
                # Calculate recency/importance score (if applicable)
                importance_score = self._calculate_importance_score(result)
                
                # Combine scores with weights
                final_score = (
                    semantic_score * 0.5 +           # Semantic similarity (50%)
                    text_relevance * 0.3 +           # Keyword relevance (30%)
                    quality_score * 0.15 +           # Content quality (15%)
                    importance_score * 0.05          # Importance/recency (5%)
                )
                
                result["reranked_score"] = final_score
                result["semantic_score"] = semantic_score
                result["text_relevance"] = text_relevance
                result["quality_score"] = quality_score
                result["importance_score"] = importance_score
            
            # Sort by reranked score
            results.sort(key=lambda x: x["reranked_score"], reverse=True)
            
            # Update the main score to the reranked score
            for result in results:
                result["score"] = result["reranked_score"]
            
            return results
            
        except Exception as e:
            st.warning(f"Reranking failed, using original results: {str(e)}")
            return results
    
    def _calculate_text_relevance(self, query: str, text: str) -> float:
        """Calculate text relevance based on keyword matching."""
        if not query or not text:
            return 0.0
        
        try:
            # Normalize text
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())
            
            # Calculate word overlap
            common_words = query_words.intersection(text_words)
            
            # Calculate relevance score
            if len(query_words) == 0:
                return 0.0
            
            # Jaccard similarity
            relevance = len(common_words) / len(query_words.union(text_words))
            
            # Boost for exact phrase matches
            if query.lower() in text.lower():
                relevance += 0.3
            
            # Boost for consecutive word matches
            query_phrases = query.lower().split()
            text_lower = text.lower()
            consecutive_matches = 0
            for i in range(len(query_phrases) - 1):
                phrase = f"{query_phrases[i]} {query_phrases[i+1]}"
                if phrase in text_lower:
                    consecutive_matches += 1
            
            if len(query_phrases) > 1:
                relevance += (consecutive_matches / (len(query_phrases) - 1)) * 0.2
            
            return min(relevance, 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_content_quality(self, result: Dict[str, Any]) -> float:
        """Calculate content quality score based on various factors."""
        try:
            text = result.get("text", "")
            text_length = result.get("text_length", 0)
            
            # Base quality score
            quality_score = 0.5
            
            # Length factor (prefer medium-length chunks)
            if 50 <= text_length <= 500:
                quality_score += 0.2
            elif text_length > 500:
                quality_score += 0.1
            elif text_length < 20:
                quality_score -= 0.2
            
            # Content richness (unique words)
            if text:
                unique_words = len(set(text.lower().split()))
                total_words = len(text.split())
                if total_words > 0:
                    diversity = unique_words / total_words
                    quality_score += diversity * 0.1
            
            # Processing method preference
            processing_method = result.get("processing_method", "")
            if "pymupdf" in processing_method.lower():
                quality_score += 0.1  # Prefer direct text extraction over OCR
            
            return min(quality_score, 1.0)
            
        except Exception:
            return 0.5
    
    def _calculate_importance_score(self, result: Dict[str, Any]) -> float:
        """Calculate importance score based on document characteristics."""
        try:
            filename = result.get("filename", "").lower()
            page_start = result.get("page_start", 1)
            
            # Base importance score
            importance_score = 0.5
            
            # Prefer bylaws and governing documents
            if any(keyword in filename for keyword in ["bylaw", "bylaws", "governing", "declaration", "covenant"]):
                importance_score += 0.3
            
            # Prefer early pages (often contain important information)
            if page_start <= 10:
                importance_score += 0.1
            
            # Prefer specific document types
            if "casitas del mar" in filename:
                importance_score += 0.2  # Prefer the main bylaws document
            
            return min(importance_score, 1.0)
            
        except Exception:
            return 0.5
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone index."""
        if not self.index:
            return {}
        
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": stats.namespaces
            }
        except Exception as e:
            st.error(f"Failed to get index stats: {str(e)}")
            return {}
    
    def delete_vectors_by_filename(self, filename: str) -> bool:
        """Delete all vectors associated with a specific filename."""
        if not self.index:
            return False
        
        try:
            # Query to find vectors with the filename
            query_results = self.index.query(
                vector=[0] * self.pinecone_config["dimension"],  # Dummy vector
                top_k=10000,  # Large number to get all
                include_metadata=True,
                filter={"filename": {"$eq": filename}}
            )
            
            # Delete found vectors
            if query_results.matches:
                vector_ids = [match.id for match in query_results.matches]
                self.index.delete(ids=vector_ids)
                st.success(f"Deleted {len(vector_ids)} vectors for {filename}")
                return True
            
            return False
            
        except Exception as e:
            st.error(f"Failed to delete vectors for {filename}: {str(e)}")
            return False
    
    def clear_all_vectors(self) -> bool:
        """Clear all vectors from the index."""
        if not self.index:
            return False
        
        try:
            self.index.delete(delete_all=True)
            st.success("All vectors cleared from Pinecone index!")
            return True
        except Exception as e:
            st.error(f"Failed to clear vectors: {str(e)}")
            return False
