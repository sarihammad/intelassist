import os
import logging
import argparse
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def create_directories():
    """Create necessary directories for the project."""
    directories = [
        "./data",
        "./data/vector_db",
        "./data/feedback",
        "./logs",
        "./app/ui/assets"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def create_env_file():
    """Create .env file if it doesn't exist."""
    if not os.path.exists(".env"):
        with open(".env.example", "r") as example_file:
            example_content = example_file.read()
        
        with open(".env", "w") as env_file:
            env_file.write(example_content)
        
        logger.info("Created .env file from .env.example")
    else:
        logger.info(".env file already exists")

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup IntelAssist project")
    parser.add_argument("--force", action="store_true", help="Force setup even if directories exist")
    args = parser.parse_args()
    
    logger.info("Starting IntelAssist setup")
    
    # Create directories
    create_directories()
    
    # Create .env file
    create_env_file()
    
    logger.info("Setup completed successfully")
    
    # Print instructions
    print("\n" + "="*50)
    print("IntelAssist Setup Complete!")
    print("="*50)
    print("\nTo run the API server:")
    print("  uvicorn app.main:app --reload")
    print("\nTo run the Streamlit UI:")
    print("  streamlit run app/ui/streamlit_app.py")
    print("\nMake sure to edit the .env file with your API keys and configuration.")
    print("="*50 + "\n")

if __name__ == "__main__":
    main() 