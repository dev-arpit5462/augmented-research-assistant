"""
Test script to validate the RAG application components.
Run this before deploying to ensure everything works correctly.
"""

import os
import sys
from pathlib import Path

# Set UTF-8 encoding for Windows compatibility
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'

def test_imports():
    """Test all required imports."""
    print("Testing imports...")
    
    try:
        # Core imports
        import streamlit as st
        print("[OK] Streamlit imported successfully")
        
        # LangChain imports
        from langchain_chroma import Chroma
        from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
        from langchain_core.documents import Document
        from langchain_core.prompts import ChatPromptTemplate
        print("[OK] LangChain components imported successfully")
        
        # LlamaIndex imports
        from llama_index.core import Document as LlamaIndexDocument
        from llama_index.readers.file import PDFReader, DocxReader
        from llama_index.core.node_parser import SentenceSplitter
        print("[OK] LlamaIndex components imported successfully")
        
        # Other dependencies
        import chromadb
        import google.generativeai as genai
        print("[OK] Vector database and AI libraries imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        print("Please install missing dependencies with: pip install -r requirements.txt")
        return False

def test_config():
    """Test configuration setup."""
    print("\nTesting configuration...")
    
    try:
        from config.settings import config, validate_config
        print("[OK] Configuration module loaded")
        
        # Test API key validation
        if os.getenv("GOOGLE_API_KEY"):
            print("[OK] Google API key found in environment")
        else:
            print("[WARNING] Google API key not found. Set GOOGLE_API_KEY environment variable.")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Configuration error: {e}")
        return False

def test_services():
    """Test service initialization."""
    print("\nTesting services...")
    
    try:
        from services.document_processor import DocumentProcessor
        from services.vector_store import VectorStoreService
        from core.rag_chain import RAGChain
        from utils.cache_manager import CacheManager
        
        print("[OK] All service modules imported successfully")
        
        # Test service initialization (without API calls)
        doc_processor = DocumentProcessor()
        print("[OK] Document processor initialized")
        
        cache_manager = CacheManager()
        print("[OK] Cache manager initialized")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Service initialization error: {e}")
        return False

def test_file_structure():
    """Test project file structure."""
    print("\nTesting file structure...")
    
    required_files = [
        "app.py",
        "requirements.txt",
        "README.md",
        ".env.example",
        "config/__init__.py",
        "config/settings.py",
        "core/__init__.py",
        "core/rag_chain.py",
        "services/__init__.py",
        "services/document_processor.py",
        "services/vector_store.py",
        "utils/__init__.py",
        "utils/cache_manager.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"[OK] {file_path}")
    
    if missing_files:
        print(f"[ERROR] Missing files: {missing_files}")
        return False
    
    print("[OK] All required files present")
    return True

def main():
    """Run all tests."""
    print("Running RAG Application Tests\n")
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Services", test_services)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[ERROR] {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{test_name:<20} {status}")
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("\nAll tests passed! Your RAG application is ready to run.")
        print("\nNext steps:")
        print("1. Set your GOOGLE_API_KEY environment variable")
        print("2. Run: streamlit run app.py")
        return 0
    else:
        print(f"\n{total - passed} test(s) failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
