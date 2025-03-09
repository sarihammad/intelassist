import os
import logging
from fastapi import APIRouter, HTTPException, Depends, Body
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from app.models.text import TextProcessor
from app.utils.auth import get_api_key

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Define request and response models
class TextRequest(BaseModel):
    text: str = Field(..., description="The input text to process")
    context: Optional[List[Dict[str, str]]] = Field(
        default=None, 
        description="Previous conversation context"
    )
    use_rag: bool = Field(
        default=True, 
        description="Whether to use Retrieval-Augmented Generation"
    )
    user_id: Optional[str] = Field(
        default=None, 
        description="User ID for personalization"
    )

class TextResponse(BaseModel):
    response: str = Field(..., description="The generated response")
    sources: Optional[List[Dict[str, Any]]] = Field(
        default=None, 
        description="Sources used for RAG"
    )
    confidence: float = Field(
        ..., 
        description="Confidence score of the response",
        ge=0.0,
        le=1.0
    )

# Initialize text processor
text_processor = TextProcessor()

@router.post("/process", response_model=TextResponse)
async def process_text(
    request: TextRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Process text input and generate a response.
    
    This endpoint handles text-based queries and generates responses using the configured
    language model. It can optionally use RAG to enhance responses with external knowledge.
    """
    try:
        logger.info(f"Processing text request: {request.text[:50]}...")
        
        # Process the text
        response, sources, confidence = text_processor.process(
            text=request.text,
            context=request.context,
            use_rag=request.use_rag,
            user_id=request.user_id
        )
        
        return TextResponse(
            response=response,
            sources=sources,
            confidence=confidence
        )
    
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")

@router.post("/chat")
async def chat(
    messages: List[Dict[str, str]] = Body(..., description="Chat messages"),
    use_rag: bool = Body(True, description="Whether to use RAG"),
    user_id: Optional[str] = Body(None, description="User ID for personalization"),
    api_key: str = Depends(get_api_key)
):
    """
    Chat endpoint that handles multi-turn conversations.
    
    This endpoint accepts a list of messages in the format:
    [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]
    """
    try:
        logger.info(f"Processing chat request with {len(messages)} messages")
        
        # Process the chat
        response = text_processor.chat(
            messages=messages,
            use_rag=use_rag,
            user_id=user_id
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@router.post("/summarize")
async def summarize(
    text: str = Body(..., description="Text to summarize"),
    max_length: int = Body(100, description="Maximum length of summary"),
    api_key: str = Depends(get_api_key)
):
    """
    Summarize a long text into a concise summary.
    """
    try:
        logger.info(f"Summarizing text of length {len(text)}")
        
        # Summarize the text
        summary = text_processor.summarize(
            text=text,
            max_length=max_length
        )
        
        return {"summary": summary}
    
    except Exception as e:
        logger.error(f"Error summarizing text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error summarizing text: {str(e)}") 