import os
import logging
import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Image processing model that handles image-based queries and generates responses.
    
    This class integrates with vision-language models for image understanding and generation.
    """
    
    def __init__(self):
        """Initialize the image processor with the configured vision model."""
        self.model_name = os.getenv("IMAGE_MODEL", "clip")
        logger.info(f"Initializing ImageProcessor with model: {self.model_name}")
        
        # Initialize the vision model based on configuration
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the vision model based on the configuration."""
        if self.model_name == "clip":
            self._initialize_clip_model()
        elif self.model_name == "blip-2":
            self._initialize_blip_model()
        else:
            logger.warning(f"Unknown model: {self.model_name}, falling back to CLIP")
            self.model_name = "clip"
            self._initialize_clip_model()
    
    def _initialize_clip_model(self):
        """Initialize CLIP model."""
        try:
            import torch
            from clip_interrogator import Config, Interrogator
            
            # Initialize CLIP model for image understanding
            config = Config(clip_model_name="ViT-L-14/openai")
            config.download_cache = True
            
            self.model = Interrogator(config)
            logger.info("Initialized CLIP model")
        except ImportError:
            logger.error("Failed to import CLIP libraries")
            raise
        except Exception as e:
            logger.error(f"Error initializing CLIP model: {str(e)}")
            raise
    
    def _initialize_blip_model(self):
        """Initialize BLIP-2 model."""
        try:
            import torch
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            # Initialize BLIP-2 model for image captioning and VQA
            processor_id = "Salesforce/blip2-opt-2.7b"
            model_id = "Salesforce/blip2-opt-2.7b"
            
            self.blip_processor = BlipProcessor.from_pretrained(processor_id)
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            logger.info("Initialized BLIP-2 model")
        except ImportError:
            logger.error("Failed to import BLIP libraries")
            raise
        except Exception as e:
            logger.error(f"Error initializing BLIP model: {str(e)}")
            raise
    
    def caption(
        self, 
        image_content: bytes,
        detailed: bool = False
    ) -> Tuple[str, float, List[str]]:
        """
        Generate a caption for an image.
        
        Args:
            image_content: Raw image bytes
            detailed: Whether to generate a detailed caption
            
        Returns:
            Tuple containing:
            - caption: The generated image caption
            - confidence: Confidence score of the caption
            - tags: Tags extracted from the image
        """
        logger.info("Generating image caption")
        start_time = time.time()
        
        try:
            # Load image
            image = Image.open(BytesIO(image_content)).convert('RGB')
            
            if self.model_name == "clip":
                # Generate caption using CLIP
                if detailed:
                    caption = self.model.interrogate(image)
                else:
                    caption = self.model.interrogate_fast(image)
                
                # Extract tags (simplified)
                tags = caption.split(",")
                tags = [tag.strip() for tag in tags[:5]]
                
                # Calculate confidence (simplified)
                confidence = 0.85  # In a real implementation, this would be model-specific
            
            elif self.model_name == "blip-2":
                # Generate caption using BLIP-2
                inputs = self.blip_processor(image, return_tensors="pt").to("cuda")
                
                if detailed:
                    prompt = "Describe this image in detail:"
                    inputs = self.blip_processor(image, text=prompt, return_tensors="pt").to("cuda")
                
                output = self.blip_model.generate(**inputs, max_new_tokens=100)
                caption = self.blip_processor.decode(output[0], skip_special_tokens=True)
                
                # Generate tags using a separate prompt
                tag_prompt = "List key objects and concepts in this image:"
                tag_inputs = self.blip_processor(image, text=tag_prompt, return_tensors="pt").to("cuda")
                tag_output = self.blip_model.generate(**tag_inputs, max_new_tokens=50)
                tag_text = self.blip_processor.decode(tag_output[0], skip_special_tokens=True)
                
                # Extract tags
                tags = [tag.strip() for tag in tag_text.split(",")]
                
                # Calculate confidence (simplified)
                confidence = 0.9  # In a real implementation, this would be model-specific
            
            logger.info(f"Generated caption in {time.time() - start_time:.2f}s")
            return caption, confidence, tags
            
        except Exception as e:
            logger.error(f"Error generating caption: {str(e)}")
            raise
    
    def analyze(
        self, 
        image_content: bytes
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Perform detailed analysis on an image.
        
        Args:
            image_content: Raw image bytes
            
        Returns:
            Tuple containing:
            - analysis: Dictionary with detailed analysis
            - objects: List of detected objects
        """
        logger.info("Analyzing image")
        start_time = time.time()
        
        try:
            # Load image
            image = Image.open(BytesIO(image_content)).convert('RGB')
            
            if self.model_name == "clip":
                # Generate caption
                caption = self.model.interrogate(image)
                
                # Extract analysis (simplified)
                analysis = {
                    "description": caption,
                    "style": self._extract_style(caption),
                    "mood": self._extract_mood(caption),
                    "setting": self._extract_setting(caption)
                }
                
                # Extract objects (simplified)
                objects = self._extract_objects_from_caption(caption)
            
            elif self.model_name == "blip-2":
                # Generate detailed description
                desc_prompt = "Describe this image in detail:"
                desc_inputs = self.blip_processor(image, text=desc_prompt, return_tensors="pt").to("cuda")
                desc_output = self.blip_model.generate(**desc_inputs, max_new_tokens=150)
                description = self.blip_processor.decode(desc_output[0], skip_special_tokens=True)
                
                # Generate style analysis
                style_prompt = "What is the visual style of this image?"
                style_inputs = self.blip_processor(image, text=style_prompt, return_tensors="pt").to("cuda")
                style_output = self.blip_model.generate(**style_inputs, max_new_tokens=30)
                style = self.blip_processor.decode(style_output[0], skip_special_tokens=True)
                
                # Generate object detection
                obj_prompt = "List all objects visible in this image:"
                obj_inputs = self.blip_processor(image, text=obj_prompt, return_tensors="pt").to("cuda")
                obj_output = self.blip_model.generate(**obj_inputs, max_new_tokens=100)
                obj_text = self.blip_processor.decode(obj_output[0], skip_special_tokens=True)
                
                # Extract analysis
                analysis = {
                    "description": description,
                    "style": style,
                    "mood": "Not available",  # Would require additional prompting
                    "setting": "Not available"  # Would require additional prompting
                }
                
                # Extract objects
                objects = [{"name": obj.strip(), "confidence": 0.8} for obj in obj_text.split(",")]
            
            logger.info(f"Analyzed image in {time.time() - start_time:.2f}s")
            return analysis, objects
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            raise
    
    def _extract_style(self, caption: str) -> str:
        """Extract the visual style from a caption (simplified)."""
        style_keywords = {
            "painting": "painting",
            "photo": "photograph",
            "drawing": "drawing",
            "sketch": "sketch",
            "digital art": "digital art",
            "illustration": "illustration",
            "cartoon": "cartoon",
            "anime": "anime",
            "3d": "3D rendering"
        }
        
        for keyword, style in style_keywords.items():
            if keyword in caption.lower():
                return style
        
        return "photograph"  # Default
    
    def _extract_mood(self, caption: str) -> str:
        """Extract the mood from a caption (simplified)."""
        mood_keywords = {
            "happy": "happy",
            "sad": "sad",
            "angry": "angry",
            "peaceful": "peaceful",
            "serene": "serene",
            "dramatic": "dramatic",
            "tense": "tense",
            "mysterious": "mysterious",
            "romantic": "romantic",
            "gloomy": "gloomy",
            "bright": "bright",
            "dark": "dark"
        }
        
        for keyword, mood in mood_keywords.items():
            if keyword in caption.lower():
                return mood
        
        return "neutral"  # Default
    
    def _extract_setting(self, caption: str) -> str:
        """Extract the setting from a caption (simplified)."""
        setting_keywords = {
            "indoor": "indoor",
            "outdoor": "outdoor",
            "nature": "nature",
            "urban": "urban",
            "rural": "rural",
            "city": "city",
            "beach": "beach",
            "mountain": "mountain",
            "forest": "forest",
            "desert": "desert",
            "studio": "studio"
        }
        
        for keyword, setting in setting_keywords.items():
            if keyword in caption.lower():
                return setting
        
        return "unknown"  # Default
    
    def _extract_objects_from_caption(self, caption: str) -> List[Dict[str, Any]]:
        """Extract objects from a caption (simplified)."""
        # Split caption by commas and clean up
        parts = [part.strip() for part in caption.split(",")]
        
        # Extract nouns as objects (very simplified)
        objects = []
        for part in parts:
            words = part.split()
            if len(words) > 0:
                # Assume the last word might be a noun
                obj_name = words[-1]
                objects.append({
                    "name": obj_name,
                    "confidence": 0.7  # Simplified confidence
                })
        
        return objects[:10]  # Limit to 10 objects
    
    def text_to_image(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512
    ) -> bytes:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text prompt for image generation
            width: Width of the generated image
            height: Height of the generated image
            
        Returns:
            Raw image bytes
        """
        logger.info(f"Generating image from text: {prompt[:50]}...")
        
        try:
            # For demonstration purposes, we'll use a placeholder
            # In a real implementation, this would use a text-to-image model like Stable Diffusion
            
            # Create a placeholder image
            image = Image.new('RGB', (width, height), color='white')
            
            # Add text to the image
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(image)
            
            # Draw the prompt text
            draw.text((10, 10), f"Prompt: {prompt}", fill='black')
            draw.text((width//2, height//2), "Image would be generated here", fill='black')
            
            # Convert to bytes
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            
            return image_bytes
            
        except Exception as e:
            logger.error(f"Error generating image from text: {str(e)}")
            raise
    
    def process_multimodal(
        self,
        image_content: bytes,
        text: str
    ) -> str:
        """
        Process a multimodal query with both image and text.
        
        Args:
            image_content: Raw image bytes
            text: Text query about the image
            
        Returns:
            Response text
        """
        logger.info(f"Processing multimodal query: {text[:50]}...")
        
        try:
            # Load image
            image = Image.open(BytesIO(image_content)).convert('RGB')
            
            if self.model_name == "blip-2":
                # Use BLIP-2 for visual question answering
                inputs = self.blip_processor(image, text=text, return_tensors="pt").to("cuda")
                output = self.blip_model.generate(**inputs, max_new_tokens=100)
                response = self.blip_processor.decode(output[0], skip_special_tokens=True)
            
            else:
                # For CLIP, we'll first generate a caption and then use a text model
                caption, _, _ = self.caption(image_content, detailed=True)
                
                # Use the text processor to answer the question based on the caption
                from app.models.text import TextProcessor
                text_processor = TextProcessor()
                
                prompt = (
                    f"I have an image with the following description: {caption}\n\n"
                    f"Question about the image: {text}\n\n"
                    f"Answer:"
                )
                
                response, _, _ = text_processor.process(prompt, use_rag=False)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing multimodal query: {str(e)}")
            raise 