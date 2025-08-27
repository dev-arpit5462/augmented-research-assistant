"""Vector store service using ChromaDB with LangChain integration."""

import os
import hashlib
from typing import List, Optional, Dict, Any
import streamlit as st
from pathlib import Path

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document as LangChainDocument
from llama_index.core import Document as LlamaIndexDocument

from config.settings import config


class VectorStoreService:
    """Manages vector storage and retrieval using ChromaDB via LangChain."""
    
    def __init__(self, user_session_id: str = None):
        self.embeddings = None
        self.vector_store = None
        self.user_session_id = user_session_id
        self._initialize_embeddings()
        self._initialize_vector_store()
    
    def _initialize_embeddings(self):
        """Initialize Google Gemini embeddings."""
        try:
            if not config.GOOGLE_API_KEY:
                print("Google API key not found. Please set GOOGLE_API_KEY environment variable.")
                return
            
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=config.GEMINI_EMBEDDING_MODEL,
                google_api_key=config.GOOGLE_API_KEY
            )
            
        except Exception as e:
            print(f"Error initializing embeddings: {str(e)}")
    
    def _initialize_vector_store(self):
        """Initialize ChromaDB vector store with complete session isolation."""
        try:
            if not self.embeddings:
                return
            
            # Ensure persist directory exists
            persist_dir = Path(config.CHROMA_PERSIST_DIRECTORY)
            persist_dir.mkdir(exist_ok=True)
            
            # Use user-specific collection name if session ID is provided
            collection_name = (config.get_user_collection_name(self.user_session_id) 
                             if self.user_session_id 
                             else config.COLLECTION_NAME)
            
            # Create user-specific subdirectory for complete isolation
            if self.user_session_id:
                user_persist_dir = persist_dir / self.user_session_id
                user_persist_dir.mkdir(exist_ok=True)
                persist_directory = str(user_persist_dir)
            else:
                persist_directory = str(persist_dir)
            
            # Force creation of separate ChromaDB client for each session
            import chromadb
            from chromadb.config import Settings
            
            # Create isolated client with session-specific settings
            # Streamlit Cloud compatibility fixes
            client_settings = Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
            
            # Create a new client instance for this session
            # Handle Streamlit Cloud deployment environment
            try:
                chroma_client = chromadb.PersistentClient(
                    path=persist_directory,
                    settings=client_settings
                )
            except Exception as e:
                # Fallback for deployment environments
                print(f"PersistentClient failed, using HttpClient: {e}")
                chroma_client = chromadb.Client(client_settings)
            
            # Initialize Chroma with the isolated client
            self.vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                client=chroma_client,
                persist_directory=persist_directory
            )
            
        except Exception as e:
            print(f"Error initializing vector store: {str(e)}")
    
    def add_documents(self, documents: List[LlamaIndexDocument]) -> bool:
        """Add LlamaIndex documents to the vector store."""
        try:
            if not self.vector_store or not documents:
                return False
            
            # Convert LlamaIndex documents to LangChain documents
            langchain_docs = []
            for doc in documents:
                langchain_doc = LangChainDocument(
                    page_content=doc.text,
                    metadata=doc.metadata or {}
                )
                langchain_docs.append(langchain_doc)
            
            # Add documents to vector store
            self.vector_store.add_documents(langchain_docs)
            
            return True
            
        except Exception as e:
            print(f"Error adding documents to vector store: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = None) -> List[LangChainDocument]:
        """Perform similarity search and return relevant documents."""
        try:
            if not self.vector_store:
                return []
            
            k = k or config.TOP_K_RETRIEVAL
            results = self.vector_store.similarity_search(query, k=k)
            
            return results
            
        except Exception as e:
            print(f"Error performing similarity search: {str(e)}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = None) -> List[tuple]:
        """Perform similarity search with relevance scores."""
        try:
            if not self.vector_store:
                return []
            
            k = k or config.TOP_K_RETRIEVAL
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            return results
            
        except Exception as e:
            print(f"Error performing similarity search with scores: {str(e)}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection."""
        try:
            if not self.vector_store:
                return {"document_count": 0, "collection_name": config.COLLECTION_NAME}
            
            # Get collection
            collection = self.vector_store._collection
            document_count = collection.count()
            
            collection_name = (config.get_user_collection_name(self.user_session_id) 
                             if self.user_session_id 
                             else config.COLLECTION_NAME)
            
            return {
                "document_count": document_count,
                "collection_name": collection_name,
                "persist_directory": config.CHROMA_PERSIST_DIRECTORY
            }
            
        except Exception as e:
            print(f"Error getting collection info: {str(e)}")
            return {"document_count": 0, "collection_name": config.COLLECTION_NAME}
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        try:
            if not self.vector_store:
                return False
            
            # Delete the collection and recreate it
            self.vector_store.delete_collection()
            self._initialize_vector_store()
            
            return True
            
        except Exception as e:
            print(f"Error clearing collection: {str(e)}")
            return False
    
    def document_exists(self, document_hash: str) -> bool:
        """Check if a document with the given hash already exists."""
        try:
            if not self.vector_store:
                return False
            
            # Search for documents with the specific hash
            results = self.vector_store.similarity_search(
                query="",  # Empty query to get any documents
                k=1000,    # Large k to check all documents
                filter={"document_hash": document_hash}
            )
            
            return len(results) > 0
            
        except Exception as e:
            # If filtering fails, assume document doesn't exist
            return False
    
    def generate_document_hash(self, content: str, filename: str) -> str:
        """Generate a hash for document deduplication."""
        combined = f"{filename}:{content}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def is_initialized(self) -> bool:
        """Check if the vector store is properly initialized."""
        return self.vector_store is not None and self.embeddings is not None
    
    def close(self):
        """Properly close the vector store connection."""
        try:
            if hasattr(self.vector_store, '_client') and self.vector_store._client:
                # Reset the client to release file locks
                self.vector_store._client.reset()
        except Exception:
            pass  # Ignore errors during cleanup
