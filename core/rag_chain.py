"""RAG chain implementation using LangChain orchestration and Gemini API."""

from typing import List, Dict, Any, Optional
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from config.settings import config
from services.vector_store import VectorStoreService
from utils.cache_manager import CacheManager


class RAGChain:
    """RAG chain orchestrator using LangChain and Gemini API."""
    
    def __init__(self, vector_store: VectorStoreService, cache_manager: CacheManager = None):
        self.vector_store = vector_store
        self.cache_manager = cache_manager
        self.llm = None
        self.chain = None
        self._initialize_llm()
        self._setup_chain()
    
    def _initialize_llm(self):
        """Initialize Google Gemini LLM."""
        try:
            if not config.GOOGLE_API_KEY:
                st.error("Google API key not found.")
                return
            
            self.llm = ChatGoogleGenerativeAI(
                model=config.GEMINI_CHAT_MODEL,
                google_api_key=config.GOOGLE_API_KEY,
                temperature=config.TEMPERATURE,
                max_output_tokens=config.MAX_TOKENS
            )
            
        except Exception as e:
            st.error(f"Error initializing LLM: {str(e)}")
    
    def _setup_chain(self):
        """Set up the RAG chain with prompt template."""
        if not self.llm:
            return
        
        # Create prompt template
        prompt_template = ChatPromptTemplate.from_template("""
You are an intelligent research assistant. Answer the question based ONLY on the provided context from the user's documents.

IMPORTANT RULES:
1. Only use information from the provided context
2. If the answer is not in the context, respond with "I couldn't find information about this in your documents."
3. Be concise and accurate
4. Cite specific parts of the context when possible
5. Do not make assumptions or add information not in the context

Context from documents:
{context}

Question: {question}

Answer:""")
        
        # Create the chain
        self.chain = (
            {"context": lambda x: self._format_docs(x["context"]), "question": lambda x: x["question"]}
            | prompt_template
            | self.llm
            | StrOutputParser()
        )
    
    def _format_docs(self, docs: List[Document]) -> str:
        """Format retrieved documents for the prompt."""
        if not docs:
            return "No relevant documents found."
        
        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source_file', 'Unknown')
            content = doc.page_content.strip()
            formatted_docs.append(f"[Source {i}: {source}]\n{content}")
        
        return "\n\n".join(formatted_docs)
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process a query through the RAG pipeline."""
        try:
            if not self.chain or not self.vector_store.is_initialized():
                return {
                    "answer": "System not properly initialized. Please check your API key and try again.",
                    "sources": [],
                    "error": True
                }
            
            # Check cache first if cache manager is available
            if self.cache_manager:
                collection_info = self.vector_store.get_collection_info()
                doc_hash = str(collection_info.get("document_count", 0))
                cached_result = self.cache_manager.get_cached_query(question, doc_hash)
                if cached_result:
                    return cached_result
            
            # Retrieve relevant documents
            docs_with_scores = self.vector_store.similarity_search_with_score(
                question, k=config.TOP_K_RETRIEVAL
            )
            
            if not docs_with_scores:
                result = {
                    "answer": "I couldn't find information about this in your documents.",
                    "sources": [],
                    "error": False
                }
                # Cache the result
                if self.cache_manager:
                    collection_info = self.vector_store.get_collection_info()
                    doc_hash = str(collection_info.get("document_count", 0))
                    self.cache_manager.cache_query(question, result, doc_hash)
                return result
            
            # Extract documents and scores
            docs = [doc for doc, score in docs_with_scores]
            scores = [score for doc, score in docs_with_scores]
            
            # Generate answer using the chain
            answer = self.chain.invoke({"question": question, "context": docs})
            
            # Prepare sources information
            sources = []
            for i, (doc, score) in enumerate(docs_with_scores):
                source_info = {
                    "source_file": doc.metadata.get('source_file', 'Unknown'),
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "relevance_score": float(score),
                    "chunk_id": doc.metadata.get('chunk_id', i)
                }
                sources.append(source_info)
            
            result = {
                "answer": answer,
                "sources": sources,
                "error": False
            }
            
            # Cache the result
            if self.cache_manager:
                collection_info = self.vector_store.get_collection_info()
                doc_hash = str(collection_info.get("document_count", 0))
                self.cache_manager.cache_query(question, result, doc_hash)
            
            return result
            
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            return {
                "answer": "An error occurred while processing your question. Please try again.",
                "sources": [],
                "error": True
            }
    
    def is_initialized(self) -> bool:
        """Check if the RAG chain is properly initialized."""
        return self.chain is not None and self.llm is not None
