import os
import logging
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import base64
from io import BytesIO

from app.models.image import ImageProcessor
from app.utils.auth import get_api_key

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Define response models
class ImageCaptionResponse(BaseModel):
    caption: str = Field(..., description="Generated caption for the image")
    confidence: float = Field(
        ..., 
        description="Confidence score of the caption",
        ge=0.0,
        le=1.0
    )
    tags: List[str] = Field(..., description="Tags extracted from the image")

class ImageAnalysisResponse(BaseModel):
    analysis: Dict[str, Any] = Field(..., description="Detailed analysis of the image")
    objects: List[Dict[str, Any]] = Field(..., description="Objects detected in the image")

# Initialize image processor
image_processor = ImageProcessor()

@router.post("/caption", response_model=ImageCaptionResponse)
async def caption_image(
    image: UploadFile = File(..., description="Image to caption"),
    detailed: bool = Form(False, description="Whether to generate a detailed caption"),
    api_key: str = Depends(get_api_key)
):
    """
    Generate a caption for an uploaded image.
    
    This endpoint uses vision-language models to generate descriptive captions for images.
    """
    try:
        logger.info(f"Processing image caption request for {image.filename}")
        
        # Read image content
        image_content = await image.read()
        
        # Process the image
        caption, confidence, tags = image_processor.caption(
            image_content=image_content,
            detailed=detailed
        )
        
        return ImageCaptionResponse(
            caption=caption,
            confidence=confidence,
            tags=tags
        )
    
    except Exception as e:
        logger.error(f"Error captioning image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error captioning image: {str(e)}")

@router.post("/analyze", response_model=ImageAnalysisResponse)
async def analyze_image(
    image: UploadFile = File(..., description="Image to analyze"),
    api_key: str = Depends(get_api_key)
):
    """
    Perform detailed analysis on an uploaded image.
    
    This endpoint detects objects, scenes, and other visual elements in the image.
    """
    try:
        logger.info(f"Processing image analysis request for {image.filename}")
        
        # Read image content
        image_content = await image.read()
        
        # Analyze the image
        analysis, objects = image_processor.analyze(
            image_content=image_content
        )
        
        return ImageAnalysisResponse(
            analysis=analysis,
            objects=objects
        )
    
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")

@router.post("/text-to-image")
async def text_to_image(
    prompt: str = Form(..., description="Text prompt for image generation"),
    width: int = Form(512, description="Width of the generated image"),
    height: int = Form(512, description="Height of the generated image"),
    api_key: str = Depends(get_api_key)
):
    """
    Generate an image from a text prompt.
    
    This endpoint uses text-to-image models to create images based on textual descriptions.
    """
    try:
        logger.info(f"Processing text-to-image request: {prompt[:50]}...")
        
        # Generate the image
        image_bytes = image_processor.text_to_image(
            prompt=prompt,
            width=width,
            height=height
        )
        
        # Convert to base64 for response
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        
        return {
            "image": base64_image,
            "prompt": prompt,
            "width": width,
            "height": height
        }
    
    except Exception as e:
        logger.error(f"Error generating image from text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating image from text: {str(e)}")

@router.post("/multimodal")
async def process_multimodal(
    image: UploadFile = File(..., description="Image to process"),
    text: str = Form(..., description="Text query about the image"),
    api_key: str = Depends(get_api_key)
):
    """
    Process a multimodal query with both image and text.
    
    This endpoint handles queries like "What's in this image?" or "Can you describe what's happening here?"
    """
    try:
        logger.info(f"Processing multimodal request: {text[:50]}...")
        
        # Read image content
        image_content = await image.read()
        
        # Process the multimodal query
        response = image_processor.process_multimodal(
            image_content=image_content,
            text=text
        )
        
        return {"response": response}
    
    except Exception as e:
        logger.error(f"Error processing multimodal query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing multimodal query: {str(e)}") 