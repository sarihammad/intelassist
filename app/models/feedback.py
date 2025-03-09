import os
import logging
import time
import uuid
import json
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import threading
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class FeedbackSystem:
    """
    Feedback system for collecting user feedback and implementing RLHF.
    
    This class handles the storage and processing of user feedback for model improvement.
    """
    
    def __init__(self):
        """Initialize the feedback system."""
        self.feedback_path = os.getenv("FEEDBACK_STORAGE_PATH", "./data/feedback")
        self.enable_rlhf = os.getenv("ENABLE_RLHF", "True").lower() == "true"
        
        logger.info(f"Initializing FeedbackSystem at {self.feedback_path} (RLHF enabled: {self.enable_rlhf})")
        
        # Ensure the feedback directory exists
        os.makedirs(self.feedback_path, exist_ok=True)
        
        # Initialize feedback storage
        self._initialize_storage()
        
        # RLHF job status
        self.rlhf_job = {
            "status": "idle",
            "last_run": None,
            "current_job_id": None,
            "progress": 0.0,
            "error": None
        }
    
    def _initialize_storage(self):
        """Initialize the feedback storage."""
        try:
            # Path to the feedback database file
            self.db_file = os.path.join(self.feedback_path, "feedback_db.json")
            
            # Check if the database file exists
            if os.path.exists(self.db_file):
                # Load existing database
                with open(self.db_file, "r") as f:
                    self.feedback_db = json.load(f)
            else:
                # Create a new database
                self.feedback_db = {
                    "feedback": [],
                    "metadata": {
                        "created_at": time.time(),
                        "last_updated": time.time(),
                        "version": "1.0"
                    }
                }
                self._save_db()
            
            logger.info(f"Initialized feedback storage with {len(self.feedback_db['feedback'])} entries")
        except Exception as e:
            logger.error(f"Error initializing feedback storage: {str(e)}")
            raise
    
    def _save_db(self):
        """Save the feedback database to disk."""
        try:
            # Update metadata
            self.feedback_db["metadata"]["last_updated"] = time.time()
            
            # Save to file
            with open(self.db_file, "w") as f:
                json.dump(self.feedback_db, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving feedback database: {str(e)}")
            raise
    
    def store(
        self,
        response_id: str,
        rating: int,
        user_id: Optional[str] = None,
        comments: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a feedback entry.
        
        Args:
            response_id: ID of the response being rated
            rating: Rating score (1-5)
            user_id: User ID providing the feedback
            comments: Additional comments about the response
            context: Context of the interaction
            
        Returns:
            ID of the stored feedback
        """
        logger.info(f"Storing feedback for response ID: {response_id}")
        
        try:
            # Generate a unique ID
            feedback_id = str(uuid.uuid4())
            
            # Create feedback entry
            entry = {
                "id": feedback_id,
                "response_id": response_id,
                "rating": rating,
                "user_id": user_id,
                "comments": comments,
                "context": context,
                "timestamp": time.time(),
                "processed_for_rlhf": False
            }
            
            # Add to database
            self.feedback_db["feedback"].append(entry)
            
            # Save the updated database
            self._save_db()
            
            logger.info(f"Stored feedback with ID: {feedback_id}")
            return feedback_id
            
        except Exception as e:
            logger.error(f"Error storing feedback: {str(e)}")
            raise
    
    def batch_store(
        self,
        entries: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Store multiple feedback entries in a single batch operation.
        
        Args:
            entries: List of feedback entries to store
            
        Returns:
            List of IDs for the stored feedback
        """
        logger.info(f"Batch storing {len(entries)} feedback entries")
        
        try:
            feedback_ids = []
            
            for entry in entries:
                # Generate a unique ID
                feedback_id = str(uuid.uuid4())
                feedback_ids.append(feedback_id)
                
                # Create feedback entry
                db_entry = {
                    "id": feedback_id,
                    "response_id": entry["response_id"],
                    "rating": entry["rating"],
                    "user_id": entry.get("user_id"),
                    "comments": entry.get("comments"),
                    "context": entry.get("context"),
                    "timestamp": time.time(),
                    "processed_for_rlhf": False
                }
                
                # Add to database
                self.feedback_db["feedback"].append(db_entry)
            
            # Save the updated database
            self._save_db()
            
            logger.info(f"Batch stored {len(feedback_ids)} feedback entries")
            return feedback_ids
            
        except Exception as e:
            logger.error(f"Error batch storing feedback: {str(e)}")
            raise
    
    def get_stats(
        self,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get statistics about user feedback.
        
        Args:
            user_id: Filter stats by user ID
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            Dictionary with feedback statistics
        """
        logger.info(f"Getting feedback stats for user_id: {user_id}")
        
        try:
            # Filter feedback entries
            filtered_feedback = self.feedback_db["feedback"]
            
            # Filter by user_id if specified
            if user_id:
                filtered_feedback = [
                    f for f in filtered_feedback
                    if f.get("user_id") == user_id
                ]
            
            # Filter by date range if specified
            if start_date:
                start_timestamp = start_date.timestamp()
                filtered_feedback = [
                    f for f in filtered_feedback
                    if f.get("timestamp", 0) >= start_timestamp
                ]
            
            if end_date:
                end_timestamp = end_date.timestamp()
                filtered_feedback = [
                    f for f in filtered_feedback
                    if f.get("timestamp", 0) <= end_timestamp
                ]
            
            # Calculate statistics
            total_feedback = len(filtered_feedback)
            
            if total_feedback == 0:
                return {
                    "total_feedback": 0,
                    "average_rating": 0.0,
                    "rating_distribution": {
                        "1": 0, "2": 0, "3": 0, "4": 0, "5": 0
                    },
                    "recent_trend": {
                        "last_day": 0.0,
                        "last_week": 0.0,
                        "last_month": 0.0
                    }
                }
            
            # Calculate average rating
            ratings = [f.get("rating", 0) for f in filtered_feedback]
            average_rating = sum(ratings) / len(ratings)
            
            # Calculate rating distribution
            rating_distribution = {
                "1": 0, "2": 0, "3": 0, "4": 0, "5": 0
            }
            for rating in ratings:
                if 1 <= rating <= 5:
                    rating_distribution[str(rating)] += 1
            
            # Calculate recent trends
            now = time.time()
            day_ago = now - (24 * 60 * 60)
            week_ago = now - (7 * 24 * 60 * 60)
            month_ago = now - (30 * 24 * 60 * 60)
            
            last_day_ratings = [
                f.get("rating", 0) for f in filtered_feedback
                if f.get("timestamp", 0) >= day_ago
            ]
            last_week_ratings = [
                f.get("rating", 0) for f in filtered_feedback
                if f.get("timestamp", 0) >= week_ago
            ]
            last_month_ratings = [
                f.get("rating", 0) for f in filtered_feedback
                if f.get("timestamp", 0) >= month_ago
            ]
            
            recent_trend = {
                "last_day": sum(last_day_ratings) / len(last_day_ratings) if last_day_ratings else 0.0,
                "last_week": sum(last_week_ratings) / len(last_week_ratings) if last_week_ratings else 0.0,
                "last_month": sum(last_month_ratings) / len(last_month_ratings) if last_month_ratings else 0.0
            }
            
            stats = {
                "total_feedback": total_feedback,
                "average_rating": average_rating,
                "rating_distribution": rating_distribution,
                "recent_trend": recent_trend
            }
            
            logger.info(f"Feedback stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting feedback stats: {str(e)}")
            raise
    
    def get_rlhf_status(self) -> Dict[str, Any]:
        """
        Get the status of the RLHF training process.
        
        Returns:
            Dictionary with RLHF status information
        """
        logger.info("Getting RLHF status")
        
        try:
            # Count unprocessed feedback
            unprocessed_count = len([
                f for f in self.feedback_db["feedback"]
                if not f.get("processed_for_rlhf", False)
            ])
            
            status = {
                **self.rlhf_job,
                "unprocessed_feedback_count": unprocessed_count,
                "rlhf_enabled": self.enable_rlhf
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting RLHF status: {str(e)}")
            raise
    
    def trigger_rlhf_update(self, force: bool = False) -> Optional[str]:
        """
        Trigger an RLHF update.
        
        Args:
            force: Whether to force an update even if one is already running
            
        Returns:
            Job ID if an update was triggered, None otherwise
        """
        logger.info(f"Triggering RLHF update (force: {force})")
        
        try:
            # Check if RLHF is enabled
            if not self.enable_rlhf:
                logger.warning("RLHF is disabled, not triggering update")
                return None
            
            # Check if an update is already running
            if self.rlhf_job["status"] == "running" and not force:
                logger.warning("RLHF update already running, not triggering another")
                return self.rlhf_job["current_job_id"]
            
            # Count unprocessed feedback
            unprocessed_feedback = [
                f for f in self.feedback_db["feedback"]
                if not f.get("processed_for_rlhf", False)
            ]
            
            if not unprocessed_feedback and not force:
                logger.info("No unprocessed feedback, not triggering RLHF update")
                return None
            
            # Generate a job ID
            job_id = str(uuid.uuid4())
            
            # Update job status
            self.rlhf_job = {
                "status": "running",
                "last_run": time.time(),
                "current_job_id": job_id,
                "progress": 0.0,
                "error": None
            }
            
            # Start RLHF update in a background thread
            thread = threading.Thread(
                target=self._run_rlhf_update,
                args=(job_id, unprocessed_feedback)
            )
            thread.daemon = True
            thread.start()
            
            logger.info(f"Triggered RLHF update with job ID: {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Error triggering RLHF update: {str(e)}")
            self.rlhf_job["status"] = "error"
            self.rlhf_job["error"] = str(e)
            raise
    
    def _run_rlhf_update(self, job_id: str, feedback_entries: List[Dict[str, Any]]):
        """
        Run the RLHF update process.
        
        Args:
            job_id: ID of the RLHF job
            feedback_entries: List of feedback entries to process
        """
        logger.info(f"Running RLHF update for job ID: {job_id}")
        
        try:
            # Simulate RLHF update process
            # In a real implementation, this would:
            # 1. Prepare training data from feedback
            # 2. Fine-tune the model using PPO
            # 3. Evaluate the fine-tuned model
            # 4. Deploy the updated model
            
            total_steps = 5
            
            # Step 1: Prepare training data
            logger.info("RLHF Step 1: Preparing training data")
            time.sleep(2)  # Simulate processing time
            self.rlhf_job["progress"] = 1 / total_steps
            
            # Step 2: Prepare reward model
            logger.info("RLHF Step 2: Preparing reward model")
            time.sleep(2)  # Simulate processing time
            self.rlhf_job["progress"] = 2 / total_steps
            
            # Step 3: Run PPO training
            logger.info("RLHF Step 3: Running PPO training")
            time.sleep(3)  # Simulate processing time
            self.rlhf_job["progress"] = 3 / total_steps
            
            # Step 4: Evaluate model
            logger.info("RLHF Step 4: Evaluating model")
            time.sleep(2)  # Simulate processing time
            self.rlhf_job["progress"] = 4 / total_steps
            
            # Step 5: Deploy model
            logger.info("RLHF Step 5: Deploying model")
            time.sleep(1)  # Simulate processing time
            
            # Mark feedback as processed
            for entry in feedback_entries:
                entry["processed_for_rlhf"] = True
            
            # Save the updated database
            self._save_db()
            
            # Update job status
            self.rlhf_job = {
                "status": "completed",
                "last_run": time.time(),
                "current_job_id": job_id,
                "progress": 1.0,
                "error": None
            }
            
            logger.info(f"Completed RLHF update for job ID: {job_id}")
            
        except Exception as e:
            logger.error(f"Error in RLHF update: {str(e)}")
            self.rlhf_job = {
                "status": "error",
                "last_run": time.time(),
                "current_job_id": job_id,
                "progress": self.rlhf_job["progress"],
                "error": str(e)
            } 