import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import Union, List
import logging

class ModelCache:
    """Handles caching and loading of sentence transformer models with fallback options."""
    
    def __init__(self):
        self.model = None
        self.tfidf_vectorizer = None
        self.model_name = "all-MiniLM-L6-v2"
        self.cache_dir = os.path.expanduser("~/.cache/huggingface/transformers/")
        
    def load_model(self) -> bool:
        """Load the sentence transformer model with error handling."""
        try:
            if self.model is None:
                with st.spinner("Loading sentence transformer model..."):
                    # Try to load the model with more detailed error handling
                    try:
                        self.model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir)
                        # Test the model with a simple encoding
                        test_embedding = self.model.encode(["test"], convert_to_numpy=True)
                        if test_embedding.shape[1] != 384:
                            raise ValueError(f"Model dimensions mismatch: expected 384, got {test_embedding.shape[1]}")
                        st.success("Model loaded successfully!")
                        return True
                    except Exception as model_error:
                        st.error(f"Model loading failed: {str(model_error)}")
                        st.info("This might be due to network issues or insufficient memory.")
                        st.info("The application will use TF-IDF fallback for embeddings.")
                        return False
        except Exception as e:
            st.warning(f"Failed to load sentence transformer model: {str(e)}")
            st.info("Falling back to TF-IDF vectorizer...")
            return False
    
    def _ensure_model_loaded(self) -> bool:
        """Ensure the model is loaded, load it if necessary."""
        if self.model is None:
            return self.load_model()
        return True
    
    def get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Get embeddings for text(s) using the loaded model or TF-IDF fallback."""
        if isinstance(texts, str):
            texts = [texts]
        
        # Try to load the model if it's not loaded yet
        if self.model is None:
            self._ensure_model_loaded()
        
        if self.model is not None:
            try:
                return self.model.encode(texts, convert_to_numpy=True)
            except Exception as e:
                st.warning(f"Error with sentence transformer: {str(e)}")
                return self._get_tfidf_embeddings(texts)
        else:
            return self._get_tfidf_embeddings(texts)
    
    def _get_tfidf_embeddings(self, texts: List[str]) -> np.ndarray:
        """Fallback to TF-IDF vectorizer for embeddings."""
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=384,
                stop_words='english',
                ngram_range=(1, 2)
            )
            # Fit on the texts
            self.tfidf_vectorizer.fit(texts)
        
        # Transform and pad/truncate to 384 dimensions
        tfidf_matrix = self.tfidf_vectorizer.transform(texts).toarray()
        
        # Pad or truncate to 384 dimensions
        if tfidf_matrix.shape[1] < 384:
            # Pad with zeros
            padded = np.zeros((tfidf_matrix.shape[0], 384))
            padded[:, :tfidf_matrix.shape[1]] = tfidf_matrix
            return padded
        else:
            # Truncate to 384 dimensions
            return tfidf_matrix[:, :384]
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if self.model is not None:
            return {
                "type": "sentence_transformer",
                "name": self.model_name,
                "dimensions": 384
            }
        else:
            return {
                "type": "tfidf_fallback",
                "dimensions": 384
            }
