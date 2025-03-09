import os
import sys
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestSetup(unittest.TestCase):
    """Test the basic setup of the IntelAssist project."""
    
    def test_imports(self):
        """Test that all required modules can be imported."""
        # Core modules
        import fastapi
        import uvicorn
        import pydantic
        import dotenv
        
        # LLM and NLP
        import transformers
        import sentence_transformers
        import langchain
        
        # Vector database (at least one should be available)
        try:
            import chromadb
        except ImportError:
            try:
                import faiss
            except ImportError:
                self.fail("Neither ChromaDB nor FAISS is available")
        
        # Web UI
        import streamlit
        
        # Project modules
        from app.main import app
        
        self.assertTrue(True, "All imports successful")
    
    def test_directories(self):
        """Test that all required directories exist."""
        required_dirs = [
            "app",
            "app/api",
            "app/models",
            "app/utils",
            "app/ui"
        ]
        
        for directory in required_dirs:
            self.assertTrue(
                os.path.isdir(os.path.join(os.path.dirname(__file__), '..', directory)),
                f"Directory {directory} does not exist"
            )
    
    def test_files(self):
        """Test that all required files exist."""
        required_files = [
            "requirements.txt",
            "README.md",
            ".env.example",
            "app/main.py",
            "app/ui/streamlit_app.py"
        ]
        
        for file in required_files:
            self.assertTrue(
                os.path.isfile(os.path.join(os.path.dirname(__file__), '..', file)),
                f"File {file} does not exist"
            )

if __name__ == '__main__':
    unittest.main() 