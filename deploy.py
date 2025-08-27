"""
Deployment helper script for the RAG application.
Handles environment setup and validation before deployment.
"""

import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def setup_environment():
    """Setup environment variables."""
    print("🔧 Setting up environment...")
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  GOOGLE_API_KEY not found in environment")
        print("Please set it using one of these methods:")
        print("1. export GOOGLE_API_KEY='your_key_here'")
        print("2. Create .env file with GOOGLE_API_KEY=your_key_here")
        print("3. Set it in your deployment platform's secrets")
        return False
    
    print("✅ Environment variables configured")
    return True

def create_directories():
    """Create necessary directories."""
    directories = [".cache", "chroma_db", ".streamlit"]
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
    print("✅ Directories created")
    return True

def run_tests():
    """Run application tests."""
    print("🧪 Running tests...")
    try:
        result = subprocess.run([sys.executable, "test_app.py"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ All tests passed")
            return True
        else:
            print("❌ Tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def start_app():
    """Start the Streamlit application."""
    print("🚀 Starting Streamlit application...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped")
    except Exception as e:
        print(f"❌ Error starting app: {e}")

def main():
    """Main deployment function."""
    print("🔍 Augmented Research Assistant - Deployment Setup")
    print("="*60)
    
    steps = [
        ("Python Version", check_python_version),
        ("Dependencies", install_dependencies),
        ("Environment", setup_environment),
        ("Directories", create_directories),
        ("Tests", run_tests)
    ]
    
    for step_name, step_func in steps:
        print(f"\n{step_name}:")
        if not step_func():
            print(f"❌ Deployment failed at: {step_name}")
            return 1
    
    print("\n🎉 Deployment setup complete!")
    print("Starting the application...")
    start_app()
    return 0

if __name__ == "__main__":
    sys.exit(main())
