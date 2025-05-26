# router_adapter.py  ─ the only networking code you need
import os, asyncio, json, time
from datetime import datetime
from pathlib import Path
from openai import AsyncOpenAI, RateLimitError, APIError, InternalServerError, Timeout
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt,
    retry_if_exception_type,
    retry_if_exception_message,
    wait_exponential,
)
from typing import Dict, Any, Union, AsyncGenerator

# Initialize OpenAI client for OpenRouter
client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),  # loaded by config.py
    base_url="https://openrouter.ai/api/v1",
    timeout=120.0,  # 2 minute timeout
    max_retries=5
)

# In-memory per-run token counter (for budgeting)
TOKENS_USED: Dict[str, int] = {"prompt": 0, "completion": 0}

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Rate limiting configuration
MAX_RPM = 60  # Maximum requests per minute
TOKEN_BUCKET = asyncio.Semaphore(MAX_RPM)  # Token bucket for rate limiting
LAST_RESET = time.time()
TOKENS_AVAILABLE = MAX_RPM
MAX_TOKENS = 500  # Default max tokens per response

def set_max_rpm(rpm: int):
    """Set the maximum requests per minute rate limit."""
    global MAX_RPM, TOKENS_AVAILABLE
    MAX_RPM = rpm
    TOKENS_AVAILABLE = rpm

def set_max_tokens(tokens: int):
    """Set the maximum tokens per response."""
    global MAX_TOKENS
    MAX_TOKENS = tokens

def get_log_file() -> Path:
    """Create a new log file with timestamp for this experiment run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOGS_DIR / f"api_traffic_{timestamp}.jsonl"

# Initialize log file for this run
LOG_FILE = get_log_file()

def safe_dict(obj: Any) -> dict:
    """Convert an object to a dict safely, handling non-serializable types."""
    try:
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)
    except Exception:
        return str(obj)

async def log_api_call(
    model_id: str,
    prompt: str,
    response: Any = None,
    error: str = None,
    start_time: float = None,
    end_time: float = None
):
    """Log an API call with its response or error."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model_id,
        "prompt": prompt,
        "response": safe_dict(response) if response else None,
        "error": error,
        "duration": end_time - start_time if start_time and end_time else None,
        "tokens_used": {
            "prompt": TOKENS_USED["prompt"],
            "completion": TOKENS_USED["completion"]
        }
    }
    
    # Add response details if available
    if response:
        log_entry["response_details"] = {
            "has_choices": hasattr(response, 'choices') and response.choices is not None,
            "choices_length": len(response.choices) if hasattr(response, 'choices') and response.choices else 0,
            "has_message": (
                hasattr(response, 'choices') and 
                response.choices and 
                hasattr(response.choices[0], 'message')
            ) if hasattr(response, 'choices') and response.choices else False,
            "has_content": (
                hasattr(response, 'choices') and 
                response.choices and 
                hasattr(response.choices[0], 'message') and
                hasattr(response.choices[0].message, 'content')
            ) if hasattr(response, 'choices') and response.choices else False,
            "raw_response": str(response)
        }
    
    async with asyncio.Lock():  # Ensure thread-safe writing
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

def is_retryable_error(e: Exception) -> bool:
    """Check if an error is retryable."""
    if isinstance(e, (RateLimitError, APIError, InternalServerError, Timeout)):
        return True
    if isinstance(e, ValueError) and "No choices in response" in str(e):
        return True
    return False

async def wait_for_token():
    """Wait for a token to become available in the rate limit bucket."""
    global LAST_RESET, TOKENS_AVAILABLE
    
    current_time = time.time()
    time_passed = current_time - LAST_RESET
    
    # Refill tokens based on time passed
    if time_passed >= 60:
        TOKENS_AVAILABLE = MAX_RPM
        LAST_RESET = current_time
    elif TOKENS_AVAILABLE < MAX_RPM:
        new_tokens = int(time_passed * (MAX_RPM / 60))
        if new_tokens > 0:
            TOKENS_AVAILABLE = min(MAX_RPM, TOKENS_AVAILABLE + new_tokens)
            LAST_RESET = current_time
    
    # Wait if no tokens available
    if TOKENS_AVAILABLE <= 0:
        wait_time = 60 - time_passed
        if wait_time > 0:
            await asyncio.sleep(wait_time)
            TOKENS_AVAILABLE = MAX_RPM
            LAST_RESET = time.time()
    
    TOKENS_AVAILABLE -= 1

@retry(
    wait=wait_exponential(multiplier=4, min=4, max=30),  # Cap at 30 seconds
    stop=stop_after_attempt(5),
    retry=(
        retry_if_exception_type((RateLimitError, APIError, InternalServerError, Timeout)) |
        retry_if_exception_message(match="No choices in response")
    ),
    reraise=True
)
async def run(model_id: str, prompt: str) -> str:
    """
    Run a model with automatic routing and token tracking.
    
    Args:
        model_id: The model identifier (e.g., "anthropic/claude-3-opus-20240229")
        prompt: The input prompt
        
    Returns:
        The model's response text
        
    Raises:
        Various API exceptions that will trigger retries
    """
    # Wait for rate limit token
    await wait_for_token()
    
    # Get the appropriate API route
    route = get_model_route(model_id)
    
    # Prepare the request
    messages = [{"role": "user", "content": prompt}]
    
    # Log the start time
    start_time = time.time()
    
    try:
        # Make the API call
        response = await client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0.0,  # Deterministic output
            max_tokens=MAX_TOKENS,  # Use configured max tokens
            timeout=120.0     # 2 minute timeout
        )
        
        # Log the response before checking for choices
        end_time = time.time()
        await log_api_call(model_id, prompt, response, start_time=start_time, end_time=end_time)
        
        # Update token usage
        update_token_usage(response)
        
        # Validate response
        if not hasattr(response, "choices") or not response.choices:
            raise ValueError("No choices in response")
            
        return response.choices[0].message.content
        
    except Exception as e:
        end_time = time.time()
        await log_api_call(model_id, prompt, None, error=e, start_time=start_time, end_time=end_time)
        raise

def get_model_route(model_id: str) -> str:
    """Get the API route for a model ID."""
    if model_id.startswith("anthropic/"):
        return "anthropic"
    return "openai"

def update_token_usage(response: Any) -> None:
    """Update global token usage from response."""
    if hasattr(response, "usage"):
        if hasattr(response.usage, "prompt_tokens"):
            TOKENS_USED["prompt"] += response.usage.prompt_tokens
        if hasattr(response.usage, "completion_tokens"):
            TOKENS_USED["completion"] += response.usage.completion_tokens
