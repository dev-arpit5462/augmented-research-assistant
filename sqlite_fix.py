"""SQLite compatibility fix for Streamlit Cloud deployment."""

import sys
import os

def fix_sqlite():
    """Fix SQLite version compatibility for ChromaDB."""
    try:
        # Force pysqlite3 import before any other SQLite usage
        import pysqlite3.dbapi2 as sqlite3_new
        
        # Replace all sqlite3 references
        sys.modules['sqlite3'] = sqlite3_new
        sys.modules['sqlite3.dbapi2'] = sqlite3_new
        
        print(f"✅ SQLite upgraded to version: {sqlite3_new.sqlite_version}")
        return True
        
    except ImportError:
        print("⚠️ pysqlite3 not available, using system SQLite")
        import sqlite3
        print(f"System SQLite version: {sqlite3.sqlite_version}")
        return False
    except Exception as e:
        print(f"❌ Error fixing SQLite: {e}")
        return False

# Auto-execute on import
fix_sqlite()
