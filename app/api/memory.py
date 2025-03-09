import os
import logging
from fastapi import APIRouter, HTTPException, Depends, Body
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from app.models.memory import MemorySystem
from app.utils.auth import get_api_key

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Define request and response models
class MemoryEntry(BaseModel):
    content: str = Field(..., description="Content to store in memory")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata for the memory entry")
    user_id: Optional[str] = Field(None, description="User ID associated with the memory")

class MemorySearchRequest(BaseModel):
    query: str = Field(..., description="Query to search for in memory")
    user_id: Optional[str] = Field(None, description="User ID to filter memories by")
    limit: int = Field(5, description="Maximum number of results to return")
    min_score: float = Field(0.7, description="Minimum similarity score threshold")

class MemorySearchResponse(BaseModel):
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    total: int = Field(..., description="Total number of matching results")

# Initialize memory system
memory_system = MemorySystem()

@router.post("/store", status_code=201)
async def store_memory(
    entry: MemoryEntry,
    api_key: str = Depends(get_api_key)
):
    """
    Store a new memory entry in the vector database.
    
    This endpoint adds new information to the system's long-term memory.
    """
    try:
        logger.info(f"Storing new memory: {entry.content[:50]}...")
        
        # Store the memory
        memory_id = memory_system.store(
            content=entry.content,
            metadata=entry.metadata,
            user_id=entry.user_id
        )
        
        return {"id": memory_id, "status": "stored"}
    
    except Exception as e:
        logger.error(f"Error storing memory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error storing memory: {str(e)}")

@router.post("/search", response_model=MemorySearchResponse)
async def search_memory(
    request: MemorySearchRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Search for relevant memories based on a query.
    
    This endpoint performs semantic search in the vector database to find relevant information.
    """
    try:
        logger.info(f"Searching memory with query: {request.query[:50]}...")
        
        # Search the memory
        results, total = memory_system.search(
            query=request.query,
            user_id=request.user_id,
            limit=request.limit,
            min_score=request.min_score
        )
        
        return MemorySearchResponse(
            results=results,
            total=total
        )
    
    except Exception as e:
        logger.error(f"Error searching memory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching memory: {str(e)}")

@router.delete("/delete/{memory_id}")
async def delete_memory(
    memory_id: str,
    user_id: Optional[str] = None,
    api_key: str = Depends(get_api_key)
):
    """
    Delete a specific memory entry by ID.
    
    This endpoint removes information from the system's long-term memory.
    """
    try:
        logger.info(f"Deleting memory with ID: {memory_id}")
        
        # Delete the memory
        success = memory_system.delete(
            memory_id=memory_id,
            user_id=user_id
        )
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Memory with ID {memory_id} not found")
        
        return {"status": "deleted", "id": memory_id}
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error deleting memory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting memory: {str(e)}")

@router.post("/batch-store", status_code=201)
async def batch_store_memories(
    entries: List[MemoryEntry],
    api_key: str = Depends(get_api_key)
):
    """
    Store multiple memory entries in a single batch operation.
    
    This endpoint is more efficient for adding multiple memories at once.
    """
    try:
        logger.info(f"Batch storing {len(entries)} memories")
        
        # Store the memories
        memory_ids = memory_system.batch_store(
            entries=[
                {
                    "content": entry.content,
                    "metadata": entry.metadata,
                    "user_id": entry.user_id
                }
                for entry in entries
            ]
        )
        
        return {"ids": memory_ids, "count": len(memory_ids), "status": "stored"}
    
    except Exception as e:
        logger.error(f"Error batch storing memories: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error batch storing memories: {str(e)}")

@router.get("/stats")
async def get_memory_stats(
    user_id: Optional[str] = None,
    api_key: str = Depends(get_api_key)
):
    """
    Get statistics about the memory system.
    
    This endpoint returns information about the number of memories, size, etc.
    """
    try:
        logger.info(f"Getting memory stats for user_id: {user_id}")
        
        # Get memory stats
        stats = memory_system.get_stats(user_id=user_id)
        
        return stats
    
    except Exception as e:
        logger.error(f"Error getting memory stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting memory stats: {str(e)}") 