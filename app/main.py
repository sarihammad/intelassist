import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.getenv("LOG_FILE", "./logs/intelassist.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="IntelAssist API",
    description="API for IntelAssist - Multimodal Self-Learning AI Assistant",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Import routers
from app.api.text import router as text_router
from app.api.image import router as image_router
from app.api.speech import router as speech_router
from app.api.memory import router as memory_router
from app.api.feedback import router as feedback_router

# Include routers
app.include_router(text_router, prefix="/api/text", tags=["Text Processing"])
app.include_router(image_router, prefix="/api/image", tags=["Image Processing"])
app.include_router(speech_router, prefix="/api/speech", tags=["Speech Processing"])
app.include_router(memory_router, prefix="/api/memory", tags=["Memory System"])
app.include_router(feedback_router, prefix="/api/feedback", tags=["Feedback & RLHF"])

@app.get("/")
async def root():
    """Root endpoint that returns basic API information."""
    return {
        "name": "IntelAssist API",
        "version": "0.1.0",
        "description": "Multimodal Self-Learning AI Assistant",
        "endpoints": {
            "text": "/api/text",
            "image": "/api/image",
            "speech": "/api/speech",
            "memory": "/api/memory",
            "feedback": "/api/feedback"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("DEBUG", "True").lower() == "true"
    ) 