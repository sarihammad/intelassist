import os
import logging
import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from io import BytesIO
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class SpeechProcessor:
    """
    Speech processing model that handles speech-to-text and text-to-speech conversions.
    
    This class integrates with speech recognition and synthesis models.
    """
    
    def __init__(self):
        """Initialize the speech processor with the configured models."""
        self.stt_model_name = os.getenv("SPEECH_TO_TEXT_MODEL", "whisper")
        self.tts_model_name = os.getenv("TEXT_TO_SPEECH_MODEL", "coqui")
        
        logger.info(f"Initializing SpeechProcessor with STT: {self.stt_model_name}, TTS: {self.tts_model_name}")
        
        # Initialize the speech models based on configuration
        self._initialize_models()
        
        # Initialize text processor for voice chat
        from app.models.text import TextProcessor
        self.text_processor = TextProcessor()
    
    def _initialize_models(self):
        """Initialize the speech models based on the configuration."""
        # Initialize speech-to-text model
        if self.stt_model_name == "whisper":
            self._initialize_whisper_model()
        else:
            logger.warning(f"Unknown STT model: {self.stt_model_name}, falling back to Whisper")
            self.stt_model_name = "whisper"
            self._initialize_whisper_model()
        
        # Initialize text-to-speech model
        if self.tts_model_name == "coqui":
            self._initialize_coqui_model()
        elif self.tts_model_name == "tacotron":
            self._initialize_tacotron_model()
        else:
            logger.warning(f"Unknown TTS model: {self.tts_model_name}, falling back to Coqui")
            self.tts_model_name = "coqui"
            self._initialize_coqui_model()
    
    def _initialize_whisper_model(self):
        """Initialize Whisper model for speech-to-text."""
        try:
            import whisper
            
            # Initialize Whisper model
            self.stt_model = whisper.load_model("base")
            logger.info("Initialized Whisper model")
        except ImportError:
            logger.error("Failed to import Whisper libraries")
            raise
        except Exception as e:
            logger.error(f"Error initializing Whisper model: {str(e)}")
            raise
    
    def _initialize_coqui_model(self):
        """Initialize Coqui TTS model for text-to-speech."""
        try:
            from TTS.api import TTS
            
            # Initialize Coqui TTS model
            self.tts_model = TTS("tts_models/en/ljspeech/tacotron2-DDC")
            logger.info("Initialized Coqui TTS model")
        except ImportError:
            logger.error("Failed to import Coqui TTS libraries")
            raise
        except Exception as e:
            logger.error(f"Error initializing Coqui TTS model: {str(e)}")
            raise
    
    def _initialize_tacotron_model(self):
        """Initialize Tacotron model for text-to-speech."""
        try:
            # This is a placeholder for Tacotron initialization
            # In a real implementation, this would use a specific Tacotron library
            logger.warning("Tacotron model not implemented, falling back to Coqui")
            self._initialize_coqui_model()
        except Exception as e:
            logger.error(f"Error initializing Tacotron model: {str(e)}")
            raise
    
    def transcribe(
        self, 
        audio_content: bytes,
        language: Optional[str] = None
    ) -> Tuple[str, float, str, float]:
        """
        Transcribe speech from audio to text.
        
        Args:
            audio_content: Raw audio bytes
            language: Language code (e.g., 'en', 'fr')
            
        Returns:
            Tuple containing:
            - text: Transcribed text
            - confidence: Confidence score of the transcription
            - language: Detected language
            - duration: Duration of the audio in seconds
        """
        logger.info("Transcribing speech to text")
        start_time = time.time()
        
        try:
            # Save audio to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_content)
                temp_file_path = temp_file.name
            
            # Transcribe using Whisper
            options = {}
            if language:
                options["language"] = language
            
            result = self.stt_model.transcribe(temp_file_path, **options)
            
            # Extract results
            text = result["text"]
            confidence = result.get("confidence", 0.8)  # Simplified
            detected_language = result.get("language", "en")
            duration = result.get("duration", len(audio_content) / 16000)  # Simplified
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            logger.info(f"Transcribed speech in {time.time() - start_time:.2f}s")
            return text, confidence, detected_language, duration
            
        except Exception as e:
            logger.error(f"Error transcribing speech: {str(e)}")
            raise
    
    def synthesize(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0
    ) -> Tuple[bytes, str, float]:
        """
        Convert text to speech.
        
        Args:
            text: Text to convert to speech
            voice: Voice ID to use
            speed: Speech speed multiplier
            
        Returns:
            Tuple containing:
            - audio_bytes: Raw audio bytes
            - audio_format: Format of the audio (e.g., 'wav')
            - duration: Duration of the audio in seconds
        """
        logger.info(f"Synthesizing speech from text: {text[:50]}...")
        start_time = time.time()
        
        try:
            # Create a temporary file for the output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file_path = temp_file.name
            
            # Synthesize speech using Coqui TTS
            if self.tts_model_name == "coqui":
                # Set voice if provided
                speaker = voice if voice != "default" else None
                
                # Generate speech
                self.tts_model.tts_to_file(
                    text=text,
                    file_path=temp_file_path,
                    speaker=speaker,
                    speed=speed
                )
            
            # Read the generated audio file
            with open(temp_file_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            # Calculate duration (simplified)
            duration = len(audio_bytes) / (16000 * 2)  # Assuming 16kHz, 16-bit audio
            
            logger.info(f"Synthesized speech in {time.time() - start_time:.2f}s")
            return audio_bytes, "wav", duration
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {str(e)}")
            raise
    
    def voice_chat(
        self,
        audio_content: bytes,
        context: Optional[List[Dict[str, str]]] = None,
        use_rag: bool = True,
        user_id: Optional[str] = None
    ) -> Tuple[str, bytes, str]:
        """
        Process voice input, generate a text response, and convert it back to speech.
        
        Args:
            audio_content: Raw audio bytes with user's speech
            context: Previous conversation context
            use_rag: Whether to use RAG
            user_id: User ID for personalization
            
        Returns:
            Tuple containing:
            - text_response: The generated text response
            - audio_response: Raw audio bytes of the spoken response
            - audio_format: Format of the audio response
        """
        logger.info("Processing voice chat")
        
        try:
            # Transcribe the user's speech
            user_text, _, _, _ = self.transcribe(audio_content)
            logger.info(f"Transcribed user speech: {user_text}")
            
            # Prepare conversation context
            if context is None:
                context = []
            
            # Add the transcribed user message to context
            context.append({"role": "user", "content": user_text})
            
            # Generate text response
            response_data = self.text_processor.chat(
                messages=context,
                use_rag=use_rag,
                user_id=user_id
            )
            
            text_response = response_data["response"]
            logger.info(f"Generated text response: {text_response[:50]}...")
            
            # Convert text response to speech
            audio_bytes, audio_format, _ = self.synthesize(text_response)
            
            return text_response, audio_bytes, audio_format
            
        except Exception as e:
            logger.error(f"Error processing voice chat: {str(e)}")
            raise
    
    def detect_language(
        self,
        audio_content: bytes
    ) -> Tuple[str, float]:
        """
        Detect the language spoken in an audio file.
        
        Args:
            audio_content: Raw audio bytes
            
        Returns:
            Tuple containing:
            - language: Detected language code
            - confidence: Confidence score of the detection
        """
        logger.info("Detecting language from speech")
        
        try:
            # Save audio to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_content)
                temp_file_path = temp_file.name
            
            # Detect language using Whisper
            result = self.stt_model.transcribe(temp_file_path, task="language")
            
            # Extract results
            language = result.get("language", "en")
            confidence = result.get("language_probability", 0.8)  # Simplified
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            logger.info(f"Detected language: {language} with confidence {confidence:.2f}")
            return language, confidence
            
        except Exception as e:
            logger.error(f"Error detecting language: {str(e)}")
            raise 