import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
import streamlit as st
import tempfile
import os
from typing import List, Dict, Any, Tuple
import re
from PIL import Image
import logging

class DocumentProcessor:
    """Handles PDF document processing, text extraction, and chunking."""
    
    def __init__(self):
        self.chunk_size = 384  # tokens
        self.chunk_overlap = 50  # tokens
        
    def extract_text_from_pdf(self, pdf_file) -> Tuple[str, Dict[str, Any]]:
        """Extract text from PDF with OCR fallback for low-density PDFs."""
        metadata = {
            "filename": pdf_file.name,
            "file_size": len(pdf_file.read()),
            "pages": 0,
            "text_density": 0,
            "processing_method": "unknown"
        }
        
        # Reset file pointer
        pdf_file.seek(0)
        
        try:
            # Try PyMuPDF first
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            metadata["pages"] = len(doc)
            
            text_content = ""
            total_chars = 0
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                text_content += page_text + "\n"
                total_chars += len(page_text)
            
            doc.close()
            
            # Calculate text density
            metadata["text_density"] = total_chars / metadata["pages"] if metadata["pages"] > 0 else 0
            metadata["processing_method"] = "pymupdf"
            
            # If text density is too low, try OCR
            if metadata["text_density"] < 100:  # Less than 100 characters per page
                st.info("Low text density detected. Attempting OCR...")
                ocr_text = self._extract_text_with_ocr(pdf_file)
                if len(ocr_text) > len(text_content):
                    text_content = ocr_text
                    metadata["processing_method"] = "ocr"
                    st.success("OCR extraction successful!")
            
            return text_content.strip(), metadata
            
        except Exception as e:
            st.error(f"Error extracting text with PyMuPDF: {str(e)}")
            st.info("Attempting OCR extraction...")
            try:
                ocr_text = self._extract_text_with_ocr(pdf_file)
                metadata["processing_method"] = "ocr"
                return ocr_text.strip(), metadata
            except Exception as ocr_error:
                st.error(f"OCR extraction also failed: {str(ocr_error)}")
                return "", metadata
    
    def _extract_text_with_ocr(self, pdf_file) -> str:
        """Extract text from PDF using OCR."""
        # Reset file pointer
        pdf_file.seek(0)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_path = tmp_file.name
        
        try:
            # Convert PDF to images
            images = convert_from_path(tmp_path, dpi=300)
            
            text_content = ""
            for i, image in enumerate(images):
                # Extract text using OCR
                page_text = pytesseract.image_to_string(image, lang='eng')
                text_content += page_text + "\n"
                
                # Show progress
                st.progress((i + 1) / len(images))
            
            return text_content
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata."""
        if not text.strip():
            return []
        
        # Simple tokenization (words)
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            if len(chunk_text.strip()) < 50:  # Skip very short chunks
                continue
            
            chunk_metadata = {
                "chunk_id": f"{metadata['filename']}_{i//self.chunk_size}",
                "filename": metadata["filename"],
                "page_start": i // self.chunk_size + 1,
                "page_end": min((i + self.chunk_size) // self.chunk_size + 1, metadata["pages"]),
                "chunk_index": i // self.chunk_size,
                "text_length": len(chunk_text),
                "processing_method": metadata["processing_method"],
                "original_metadata": metadata
            }
            
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })
        
        return chunks
    
    def process_document(self, pdf_file) -> List[Dict[str, Any]]:
        """Complete document processing pipeline."""
        with st.spinner(f"Processing {pdf_file.name}..."):
            # Extract text
            text, metadata = self.extract_text_from_pdf(pdf_file)
            
            if not text:
                st.error(f"Failed to extract text from {pdf_file.name}")
                return []
            
            # Chunk text
            chunks = self.chunk_text(text, metadata)
            
            st.success(f"Successfully processed {pdf_file.name}: {len(chunks)} chunks created")
            
            return chunks
    
    def get_document_summary(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of the processed document."""
        if not chunks:
            return {}
        
        total_chunks = len(chunks)
        total_text_length = sum(chunk["metadata"]["text_length"] for chunk in chunks)
        avg_chunk_length = total_text_length / total_chunks if total_chunks > 0 else 0
        
        # Get unique filenames
        filenames = list(set(chunk["metadata"]["filename"] for chunk in chunks))
        
        return {
            "total_chunks": total_chunks,
            "total_text_length": total_text_length,
            "avg_chunk_length": avg_chunk_length,
            "filenames": filenames,
            "processing_methods": list(set(chunk["metadata"]["processing_method"] for chunk in chunks))
        }
