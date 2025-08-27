"""Configuration settings for the Augmented Research Assistant."""

import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class AppConfig:
    """Application configuration settings."""
    
    # API Keys
    GOOGLE_API_KEY: Optional[str] = None
    
    # Gemini Model Configuration
    GEMINI_CHAT_MODEL: str = "gemini-pro"
    GEMINI_EMBEDDING_MODEL: str = "models/embedding-001"
    
    # Vector Database Settings
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"
    COLLECTION_NAME: str = "documents"
    
    def get_user_collection_name(self, user_session_id: str) -> str:
        """Get user-specific collection name."""
        return f"user_{user_session_id}_docs"
    
    # Document Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_FILE_SIZE_MB: int = 10
    
    # RAG Settings
    TOP_K_RETRIEVAL: int = 5
    TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 1024
    
    # UI Settings
    PAGE_TITLE: str = "🔍 Augmented Research Assistant"
    PAGE_ICON: str = "🔍"
    LAYOUT: str = "wide"
    
    # Caching
    ENABLE_CACHING: bool = True
    CACHE_TTL: int = 3600  # 1 hour
    
    def __post_init__(self):
        """Load environment variables after initialization."""
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", self.GEMINI_CHAT_MODEL)
        self.GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", self.GEMINI_EMBEDDING_MODEL)

# Global config instance
config = AppConfig()

def validate_config() -> bool:
    """Validate that required configuration is present."""
    if not config.GOOGLE_API_KEY:
        return False
    return True

def get_supported_file_types() -> list:
    """Get list of supported file types."""
    return [".pdf", ".txt", ".docx", ".md"]
