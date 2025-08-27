"""Document processing service using LlamaIndex loaders."""

import os
import tempfile
from typing import List, Optional
from pathlib import Path
import streamlit as st

from llama_index.core import Document
from llama_index.readers.file import PDFReader, DocxReader
from llama_index.core.node_parser import SentenceSplitter

from config.settings import config


class DocumentProcessor:
    """Handles document ingestion and processing using LlamaIndex."""
    
    def __init__(self):
        self.pdf_reader = PDFReader()
        self.docx_reader = DocxReader()
        self.text_splitter = SentenceSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
    
    def process_uploaded_file(self, uploaded_file) -> Optional[List[Document]]:
        """Process an uploaded file and return LlamaIndex documents."""
        try:
            # Check file size
            if uploaded_file.size > config.MAX_FILE_SIZE_MB * 1024 * 1024:
                st.error(f"File size exceeds {config.MAX_FILE_SIZE_MB}MB limit")
                return None
            
            # Get file extension
            file_extension = Path(uploaded_file.name).suffix.lower()
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Process based on file type
                documents = self._load_document(tmp_file_path, file_extension, uploaded_file.name)
                
                if documents:
                    # Split documents into chunks
                    nodes = self.text_splitter.get_nodes_from_documents(documents)
                    
                    # Convert nodes back to documents with metadata
                    chunked_docs = []
                    for i, node in enumerate(nodes):
                        doc = Document(
                            text=node.text,
                            metadata={
                                **node.metadata,
                                "source_file": uploaded_file.name,
                                "chunk_id": i,
                                "file_type": file_extension
                            }
                        )
                        chunked_docs.append(doc)
                    
                    return chunked_docs
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
            return None
    
    def _load_document(self, file_path: str, file_extension: str, original_name: str) -> Optional[List[Document]]:
        """Load document based on file type."""
        try:
            if file_extension == ".pdf":
                documents = self.pdf_reader.load_data(file_path)
            elif file_extension == ".docx":
                documents = self.docx_reader.load_data(file_path)
            elif file_extension in [".txt", ".md"]:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents = [Document(text=content)]
            else:
                st.error(f"Unsupported file type: {file_extension}")
                return None
            
            # Add source metadata
            for doc in documents:
                doc.metadata.update({
                    "source_file": original_name,
                    "file_type": file_extension
                })
            
            return documents
            
        except Exception as e:
            st.error(f"Error loading document: {str(e)}")
            return None
    
    def get_document_stats(self, documents: List[Document]) -> dict:
        """Get statistics about processed documents."""
        if not documents:
            return {"total_chunks": 0, "total_characters": 0}
        
        total_chars = sum(len(doc.text) for doc in documents)
        return {
            "total_chunks": len(documents),
            "total_characters": total_chars,
            "avg_chunk_size": total_chars // len(documents) if documents else 0
        }
