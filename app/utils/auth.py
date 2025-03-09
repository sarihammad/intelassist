import os
import logging
from fastapi import HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Define API key header
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# In a production environment, you would store API keys securely
# This is a simplified implementation for demonstration purposes
API_KEYS = {
    "test_key": "development",  # For testing purposes
}

# Add environment API keys if available
if OPENAI_API_KEY:
    API_KEYS[OPENAI_API_KEY] = "openai"
if HUGGINGFACE_API_KEY:
    API_KEYS[HUGGINGFACE_API_KEY] = "huggingface"

async def get_api_key(api_key_header: str = Security(api_key_header)):
    """
    Validate the API key provided in the request header.
    
    This function checks if the provided API key is valid and returns it if so.
    Otherwise, it raises an HTTP 403 Forbidden exception.
    """
    if not api_key_header:
        logger.warning("API key missing in request")
        raise HTTPException(
            status_code=403,
            detail="API key is required"
        )
    
    if api_key_header in API_KEYS:
        logger.debug(f"Valid API key provided: {API_KEYS[api_key_header]}")
        return api_key_header
    
    logger.warning(f"Invalid API key provided")
    raise HTTPException(
        status_code=403,
        detail="Invalid API key"
    )

def is_admin(api_key: str = Depends(get_api_key)):
    """
    Check if the provided API key has admin privileges.
    
    This function is used for endpoints that require administrative access.
    """
    # In a real implementation, you would check against a database of user roles
    # This is a simplified implementation for demonstration purposes
    admin_keys = {key for key, role in API_KEYS.items() if role == "admin"}
    
    if api_key not in admin_keys:
        logger.warning(f"Non-admin API key used for admin endpoint")
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )
    
    return True 