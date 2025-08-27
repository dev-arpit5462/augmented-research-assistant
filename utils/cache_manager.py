"""Caching utilities for improved performance."""

import hashlib
import pickle
import os
from typing import Any, Optional
from pathlib import Path
import streamlit as st

from config.settings import config


class CacheManager:
    """Manages caching for embeddings and query results."""
    
    def __init__(self, user_session_id: str = None):
        self.user_session_id = user_session_id or "default"
        self.cache_dir = Path(".cache") / self.user_session_id
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_cache_dir = self.cache_dir / "embeddings"
        self.queries_cache_dir = self.cache_dir / "queries"
        
        # Create subdirectories
        self.embeddings_cache_dir.mkdir(exist_ok=True)
        self.queries_cache_dir.mkdir(exist_ok=True)
    
    def _generate_cache_key(self, data: str) -> str:
        """Generate a cache key from input data."""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _get_cache_path(self, cache_type: str, key: str) -> Path:
        """Get the full path for a cache file."""
        if cache_type == "embedding":
            return self.embeddings_cache_dir / f"{key}.pkl"
        elif cache_type == "query":
            return self.queries_cache_dir / f"{key}.pkl"
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")
    
    def get_cached_embedding(self, text: str) -> Optional[Any]:
        """Retrieve cached embedding for text."""
        if not config.ENABLE_CACHING:
            return None
        
        try:
            cache_key = self._generate_cache_key(text)
            cache_path = self._get_cache_path("embedding", cache_key)
            
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        except Exception:
            pass
        
        return None
    
    def cache_embedding(self, text: str, embedding: Any) -> bool:
        """Cache an embedding for text."""
        if not config.ENABLE_CACHING:
            return False
        
        try:
            cache_key = self._generate_cache_key(text)
            cache_path = self._get_cache_path("embedding", cache_key)
            
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
            
            return True
        except Exception:
            return False
    
    def get_cached_query(self, query: str, doc_hash: str = "") -> Optional[Any]:
        """Retrieve cached query result."""
        if not config.ENABLE_CACHING:
            return None
        
        try:
            cache_key = self._generate_cache_key(f"{query}:{doc_hash}")
            cache_path = self._get_cache_path("query", cache_key)
            
            if cache_path.exists():
                # Check if cache is still valid (TTL)
                import time
                cache_age = cache_path.stat().st_mtime
                current_time = time.time()
                
                if (current_time - cache_age) < config.CACHE_TTL:
                    with open(cache_path, 'rb') as f:
                        return pickle.load(f)
                else:
                    # Remove expired cache
                    cache_path.unlink()
        except Exception:
            pass
        
        return None
    
    def cache_query(self, query: str, result: Any, doc_hash: str = "") -> bool:
        """Cache a query result."""
        if not config.ENABLE_CACHING:
            return False
        
        try:
            cache_key = self._generate_cache_key(f"{query}:{doc_hash}")
            cache_path = self._get_cache_path("query", cache_key)
            
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            
            return True
        except Exception:
            return False
    
    def clear_cache(self, cache_type: Optional[str] = None) -> bool:
        """Clear cache files."""
        try:
            if cache_type == "embedding" or cache_type is None:
                for file in self.embeddings_cache_dir.glob("*.pkl"):
                    file.unlink()
            
            if cache_type == "query" or cache_type is None:
                for file in self.queries_cache_dir.glob("*.pkl"):
                    file.unlink()
            
            return True
        except Exception:
            return False
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        try:
            embedding_files = list(self.embeddings_cache_dir.glob("*.pkl"))
            query_files = list(self.queries_cache_dir.glob("*.pkl"))
            
            embedding_size = sum(f.stat().st_size for f in embedding_files)
            query_size = sum(f.stat().st_size for f in query_files)
            
            return {
                "embedding_cache_count": len(embedding_files),
                "query_cache_count": len(query_files),
                "embedding_cache_size_mb": embedding_size / (1024 * 1024),
                "query_cache_size_mb": query_size / (1024 * 1024),
                "total_size_mb": (embedding_size + query_size) / (1024 * 1024)
            }
        except Exception:
            return {
                "embedding_cache_count": 0,
                "query_cache_count": 0,
                "embedding_cache_size_mb": 0,
                "query_cache_size_mb": 0,
                "total_size_mb": 0
            }
