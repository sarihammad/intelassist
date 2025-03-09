import os
import logging
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import base64
from io import BytesIO

from app.models.speech import SpeechProcessor
from app.utils.auth import get_api_key

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Define response models
class SpeechToTextResponse(BaseModel):
    text: str = Field(..., description="Transcribed text from speech")
    confidence: float = Field(
        ..., 
        description="Confidence score of the transcription",
        ge=0.0,
        le=1.0
    )
    language: str = Field(..., description="Detected language of the speech")
    duration: float = Field(..., description="Duration of the audio in seconds")

# Initialize speech processor
speech_processor = SpeechProcessor()

@router.post("/speech-to-text", response_model=SpeechToTextResponse)
async def speech_to_text(
    audio: UploadFile = File(..., description="Audio file to transcribe"),
    language: Optional[str] = Form(None, description="Language code (e.g., 'en', 'fr')"),
    api_key: str = Depends(get_api_key)
):
    """
    Transcribe speech from an audio file to text.
    
    This endpoint uses speech recognition models to convert spoken language to text.
    """
    try:
        logger.info(f"Processing speech-to-text request for {audio.filename}")
        
        # Read audio content
        audio_content = await audio.read()
        
        # Process the audio
        text, confidence, detected_language, duration = speech_processor.transcribe(
            audio_content=audio_content,
            language=language
        )
        
        return SpeechToTextResponse(
            text=text,
            confidence=confidence,
            language=detected_language,
            duration=duration
        )
    
    except Exception as e:
        logger.error(f"Error transcribing speech: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error transcribing speech: {str(e)}")

@router.post("/text-to-speech")
async def text_to_speech(
    text: str = Form(..., description="Text to convert to speech"),
    voice: str = Form("default", description="Voice ID to use"),
    speed: float = Form(1.0, description="Speech speed multiplier"),
    api_key: str = Depends(get_api_key)
):
    """
    Convert text to speech.
    
    This endpoint uses text-to-speech models to generate natural-sounding speech from text.
    """
    try:
        logger.info(f"Processing text-to-speech request: {text[:50]}...")
        
        # Generate speech
        audio_bytes, audio_format, duration = speech_processor.synthesize(
            text=text,
            voice=voice,
            speed=speed
        )
        
        # Convert to base64 for response
        base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
        
        return {
            "audio": base64_audio,
            "format": audio_format,
            "duration": duration,
            "text": text
        }
    
    except Exception as e:
        logger.error(f"Error synthesizing speech: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error synthesizing speech: {str(e)}")

@router.post("/voice-chat")
async def voice_chat(
    audio: UploadFile = File(..., description="Audio file with user's speech"),
    context: Optional[List[Dict[str, str]]] = Form(None, description="Previous conversation context"),
    use_rag: bool = Form(True, description="Whether to use RAG"),
    user_id: Optional[str] = Form(None, description="User ID for personalization"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    api_key: str = Depends(get_api_key)
):
    """
    Process voice input, generate a text response, and convert it back to speech.
    
    This endpoint combines speech-to-text, text processing, and text-to-speech for voice conversations.
    """
    try:
        logger.info(f"Processing voice chat request")
        
        # Read audio content
        audio_content = await audio.read()
        
        # Process the voice chat
        text_response, audio_bytes, audio_format = speech_processor.voice_chat(
            audio_content=audio_content,
            context=context,
            use_rag=use_rag,
            user_id=user_id
        )
        
        # Convert to base64 for response
        base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
        
        return {
            "text_response": text_response,
            "audio_response": base64_audio,
            "format": audio_format
        }
    
    except Exception as e:
        logger.error(f"Error processing voice chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing voice chat: {str(e)}")

@router.post("/detect-language")
async def detect_language(
    audio: UploadFile = File(..., description="Audio file to detect language from"),
    api_key: str = Depends(get_api_key)
):
    """
    Detect the language spoken in an audio file.
    
    This endpoint identifies the language being spoken in the provided audio.
    """
    try:
        logger.info(f"Processing language detection request for {audio.filename}")
        
        # Read audio content
        audio_content = await audio.read()
        
        # Detect language
        language, confidence = speech_processor.detect_language(
            audio_content=audio_content
        )
        
        return {
            "language": language,
            "confidence": confidence
        }
    
    except Exception as e:
        logger.error(f"Error detecting language: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error detecting language: {str(e)}") 