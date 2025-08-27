"""SQLite compatibility fix for Streamlit Cloud deployment."""

import sys
import sqlite3

def fix_sqlite():
    """Fix SQLite version compatibility for ChromaDB."""
    try:
        # Check current SQLite version
        current_version = sqlite3.sqlite_version
        print(f"Current SQLite version: {current_version}")
        
        # ChromaDB requires SQLite >= 3.35.0
        required_version = "3.35.0"
        
        if current_version < required_version:
            print(f"SQLite version {current_version} is below required {required_version}")
            
            # Try to use pysqlite3-binary if available
            try:
                import pysqlite3.dbapi2 as sqlite3_new
                # Monkey patch sqlite3
                sys.modules['sqlite3'] = sqlite3_new
                print("Successfully patched SQLite with pysqlite3-binary")
                return True
            except ImportError:
                print("pysqlite3-binary not available")
                return False
        else:
            print(f"SQLite version {current_version} is compatible")
            return True
            
    except Exception as e:
        print(f"Error checking SQLite version: {e}")
        return False

if __name__ == "__main__":
    fix_sqlite()
