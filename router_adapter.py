# router_adapter.py  ─ the only networking code you need
import os, asyncio, json
from datetime import datetime
from pathlib import Path
from openai import AsyncOpenAI, RateLimitError, APIError, InternalServerError, Timeout
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)
from typing import Dict

# Initialize OpenAI client for OpenRouter
client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),  # loaded by config.py
    base_url="https://openrouter.ai/api/v1"
)

# In-memory per-run token counter (for budgeting)
TOKENS_USED: Dict[str, int] = {"prompt": 0, "completion": 0}

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

def get_log_file() -> Path:
    """Create a new log file with timestamp for this experiment run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOGS_DIR / f"api_traffic_{timestamp}.jsonl"

# Initialize log file for this run
LOG_FILE = get_log_file()

async def log_api_call(model_id: str, prompt: str, response: dict, error: str = None):
    """Log an API call with its response or error."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model_id,
        "prompt": prompt,
        "response": response,
        "error": error,
        "tokens_used": {
            "prompt": TOKENS_USED["prompt"],
            "completion": TOKENS_USED["completion"]
        }
    }
    
    async with asyncio.Lock():  # Ensure thread-safe writing
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

@retry(
    wait=wait_random_exponential(multiplier=1, max=120),  # Increased max wait time
    stop=stop_after_attempt(5),  # Increased retry attempts
    retry=retry_if_exception_type((RateLimitError, APIError, InternalServerError, Timeout)),
)
async def run(model_id: str, prompt: str, timeout: int = 180) -> str:  # Increased default timeout
    """
    Async call to an OpenRouter model.  Returns raw text content.
    Retries with exponential back-off on 429s & transient errors.
    """
    try:
        response = await client.chat.completions.create(
            model=model_id,
            temperature=0.7,              # for better diversity
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout,
            stream=False,
        )

        # Track token usage
        if response.usage:
            TOKENS_USED["prompt"] += int(response.usage.prompt_tokens or 0)
            TOKENS_USED["completion"] += int(response.usage.completion_tokens or 0)

        # OpenRouter always returns OpenAI format
        if not response.choices:
            raise ValueError("No choices in response")
            
        # Log successful API call
        await log_api_call(model_id, prompt, response.model_dump())
        
        return response.choices[0].message.content

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"\nError calling model {model_id}:")
        print(error_msg)
        
        # Log failed API call
        await log_api_call(model_id, prompt, None, error_msg)
        
        raise
