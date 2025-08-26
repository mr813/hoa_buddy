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
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using semantic similarity."""
        if not self.index or not query.strip():
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.model_cache.get_embeddings(query)
            
            # Search in Pinecone with new API
            search_results = self.index.query(
                vector=query_embedding[0].tolist(),
                top_k=top_k,
                include_metadata=True
            )
            
            # Format results
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
            
            return results
            
        except Exception as e:
            st.error(f"Failed to search in Pinecone: {str(e)}")
            return []
    
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
