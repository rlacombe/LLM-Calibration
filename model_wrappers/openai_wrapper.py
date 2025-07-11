from typing import Dict, Any, Optional
from openai import AsyncOpenAI, RateLimitError, APIError, InternalServerError, Timeout
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
    retry_if_exception_message,
)
from .base import BaseModelWrapper, TokenUsage

class OpenAIModelWrapper(BaseModelWrapper):
    """OpenAI model wrapper implementation."""
    
    def __init__(self, model_id: str, api_key: str, **kwargs):
        super().__init__(model_id, api_key, **kwargs)
        self.client = AsyncOpenAI(
            api_key=api_key,
            timeout=120.0,  # 2 minute timeout
            max_retries=5
        )
    
    def supports_reasoning_budget(self) -> bool:
        """OpenAI models don't support reasoning budget features."""
        return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information."""
        return {
            "provider": "openai",
            "model_id": self.model_id,
            "supports_reasoning_budget": False,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
    
    @retry(
        wait=wait_exponential(multiplier=4, min=4, max=30),
        stop=stop_after_attempt(5),
        retry=(
            retry_if_exception_type((RateLimitError, APIError, InternalServerError, Timeout)) |
            retry_if_exception_message(match="No choices in response")
        ),
        reraise=True
    )
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response using the OpenAI API."""
        messages = [{"role": "user", "content": prompt}]
        
        response = await self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs
        )
        
        # Update token usage
        if hasattr(response, "usage"):
            self.update_token_usage(
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )
        
        # Validate response
        if not hasattr(response, "choices") or not response.choices:
            raise ValueError("No choices in response")
        
        return response.choices[0].message.content 