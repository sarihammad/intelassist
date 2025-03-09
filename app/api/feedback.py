import os
import logging
from fastapi import APIRouter, HTTPException, Depends, Body
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

from app.models.feedback import FeedbackSystem
from app.utils.auth import get_api_key

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Define request and response models
class FeedbackEntry(BaseModel):
    response_id: str = Field(..., description="ID of the response being rated")
    rating: int = Field(..., description="Rating score (1-5)", ge=1, le=5)
    user_id: Optional[str] = Field(None, description="User ID providing the feedback")
    comments: Optional[str] = Field(None, description="Additional comments about the response")
    context: Optional[Dict[str, Any]] = Field(None, description="Context of the interaction")

class FeedbackStats(BaseModel):
    total_feedback: int = Field(..., description="Total number of feedback entries")
    average_rating: float = Field(..., description="Average rating score")
    rating_distribution: Dict[str, int] = Field(..., description="Distribution of ratings")
    recent_trend: Dict[str, float] = Field(..., description="Recent trend in ratings")

# Initialize feedback system
feedback_system = FeedbackSystem()

@router.post("/submit", status_code=201)
async def submit_feedback(
    feedback: FeedbackEntry,
    api_key: str = Depends(get_api_key)
):
    """
    Submit user feedback for a response.
    
    This endpoint collects user ratings and comments for RLHF.
    """
    try:
        logger.info(f"Submitting feedback for response ID: {feedback.response_id}")
        
        # Store the feedback
        feedback_id = feedback_system.store(
            response_id=feedback.response_id,
            rating=feedback.rating,
            user_id=feedback.user_id,
            comments=feedback.comments,
            context=feedback.context
        )
        
        # Trigger RLHF update if enabled
        if os.getenv("ENABLE_RLHF", "True").lower() == "true":
            feedback_system.trigger_rlhf_update()
        
        return {"id": feedback_id, "status": "submitted"}
    
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error submitting feedback: {str(e)}")

@router.get("/stats", response_model=FeedbackStats)
async def get_feedback_stats(
    user_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    api_key: str = Depends(get_api_key)
):
    """
    Get statistics about user feedback.
    
    This endpoint returns aggregated feedback statistics for analysis.
    """
    try:
        logger.info(f"Getting feedback stats for user_id: {user_id}")
        
        # Parse dates if provided
        start_datetime = datetime.fromisoformat(start_date) if start_date else None
        end_datetime = datetime.fromisoformat(end_date) if end_date else None
        
        # Get feedback stats
        stats = feedback_system.get_stats(
            user_id=user_id,
            start_date=start_datetime,
            end_date=end_datetime
        )
        
        return FeedbackStats(
            total_feedback=stats["total_feedback"],
            average_rating=stats["average_rating"],
            rating_distribution=stats["rating_distribution"],
            recent_trend=stats["recent_trend"]
        )
    
    except Exception as e:
        logger.error(f"Error getting feedback stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting feedback stats: {str(e)}")

@router.post("/batch-submit", status_code=201)
async def batch_submit_feedback(
    feedback_entries: List[FeedbackEntry],
    api_key: str = Depends(get_api_key)
):
    """
    Submit multiple feedback entries in a single batch operation.
    
    This endpoint is more efficient for adding multiple feedback entries at once.
    """
    try:
        logger.info(f"Batch submitting {len(feedback_entries)} feedback entries")
        
        # Store the feedback entries
        feedback_ids = feedback_system.batch_store(
            entries=[
                {
                    "response_id": entry.response_id,
                    "rating": entry.rating,
                    "user_id": entry.user_id,
                    "comments": entry.comments,
                    "context": entry.context
                }
                for entry in feedback_entries
            ]
        )
        
        # Trigger RLHF update if enabled
        if os.getenv("ENABLE_RLHF", "True").lower() == "true":
            feedback_system.trigger_rlhf_update()
        
        return {"ids": feedback_ids, "count": len(feedback_ids), "status": "submitted"}
    
    except Exception as e:
        logger.error(f"Error batch submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error batch submitting feedback: {str(e)}")

@router.get("/rlhf-status")
async def get_rlhf_status(
    api_key: str = Depends(get_api_key)
):
    """
    Get the status of the RLHF training process.
    
    This endpoint returns information about the current RLHF training status.
    """
    try:
        logger.info("Getting RLHF status")
        
        # Get RLHF status
        status = feedback_system.get_rlhf_status()
        
        return status
    
    except Exception as e:
        logger.error(f"Error getting RLHF status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting RLHF status: {str(e)}")

@router.post("/trigger-rlhf")
async def trigger_rlhf_update(
    api_key: str = Depends(get_api_key)
):
    """
    Manually trigger an RLHF update.
    
    This endpoint initiates the RLHF training process to update the model based on feedback.
    """
    try:
        logger.info("Manually triggering RLHF update")
        
        # Trigger RLHF update
        job_id = feedback_system.trigger_rlhf_update(force=True)
        
        return {"job_id": job_id, "status": "triggered"}
    
    except Exception as e:
        logger.error(f"Error triggering RLHF update: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error triggering RLHF update: {str(e)}") 