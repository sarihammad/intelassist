#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def run_setup():
    """Run the setup script."""
    logger.info("Running setup script")
    subprocess.run([sys.executable, "setup.py"])

def run_api():
    """Run the API server."""
    host = os.getenv("HOST", "0.0.0.0")
    port = os.getenv("PORT", "8000")
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    logger.info(f"Starting API server on {host}:{port} (debug: {debug})")
    
    cmd = [
        "uvicorn", 
        "app.main:app", 
        "--host", host, 
        "--port", port
    ]
    
    if debug:
        cmd.append("--reload")
    
    subprocess.run(cmd)

def run_ui():
    """Run the Streamlit UI."""
    logger.info("Starting Streamlit UI")
    subprocess.run(["streamlit", "run", "app/ui/streamlit_app.py"])

def run_docker():
    """Run the application using Docker Compose."""
    logger.info("Starting application with Docker Compose")
    subprocess.run(["docker-compose", "up", "--build"])

def main():
    """Main function to parse arguments and run the appropriate command."""
    parser = argparse.ArgumentParser(description="Run IntelAssist")
    parser.add_argument("command", choices=["setup", "api", "ui", "docker", "all"], 
                        help="Command to run")
    args = parser.parse_args()
    
    if args.command == "setup":
        run_setup()
    elif args.command == "api":
        run_setup()
        run_api()
    elif args.command == "ui":
        run_setup()
        run_ui()
    elif args.command == "docker":
        run_setup()
        run_docker()
    elif args.command == "all":
        run_setup()
        # Start API in a separate process
        api_process = subprocess.Popen([
            sys.executable, "-c", 
            "import run; run.run_api()"
        ])
        # Start UI in the main process
        run_ui()
        # Terminate API process when UI is closed
        api_process.terminate()

if __name__ == "__main__":
    main() 