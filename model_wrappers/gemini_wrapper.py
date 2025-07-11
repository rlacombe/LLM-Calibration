from typing import Dict, Any, Optional
import google.generativeai as genai
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)
from .base import BaseModelWrapper, TokenUsage

class GeminiModelWrapper(BaseModelWrapper):
    """Google Gemini model wrapper implementation."""
    
    def __init__(self, model_id: str, api_key: str, **kwargs):
        super().__init__(model_id, api_key, **kwargs)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_id)
    
    def supports_reasoning_budget(self) -> bool:
        """Gemini models support reasoning budget features."""
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Gemini model information."""
        return {
            "provider": "google",
            "model_id": self.model_id,
            "supports_reasoning_budget": True,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "reasoning_budget": self.reasoning_budget
        }
    
    @retry(
        wait=wait_exponential(multiplier=4, min=4, max=30),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response using the Gemini API."""
        # Configure generation parameters
        generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }
        
        # Add reasoning budget if supported and specified
        if self.reasoning_budget is not None:
            generation_config["reasoning_budget"] = self.reasoning_budget
        
        # Create chat session
        chat = self.model.start_chat()
        
        # Generate response
        response = await chat.send_message_async(
            prompt,
            generation_config=generation_config,
            **kwargs
        )
        
        # Update token usage (Gemini provides token counts in response)
        if hasattr(response, "token_count"):
            self.update_token_usage(
                response.token_count.prompt_tokens,
                response.token_count.completion_tokens
            )
        
        return response.text 