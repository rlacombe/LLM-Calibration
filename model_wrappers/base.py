from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class BaseModelWrapper(ABC):
    """Base class for all model wrappers."""
    
    def __init__(
        self,
        model_id: str,
        api_key: str,
        reasoning_budget: Optional[float] = None,
        max_tokens: int = 500,
        temperature: float = 0.0,
        **kwargs
    ):
        self.model_id = model_id
        self.api_key = api_key
        self.reasoning_budget = reasoning_budget
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.token_usage = TokenUsage()
        self.last_request_time: Optional[datetime] = None
        
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response with the given prompt and parameters.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional model-specific parameters
            
        Returns:
            The model's response text
            
        Raises:
            ModelError: If the model fails to generate a response
        """
        pass
    
    def get_token_usage(self) -> TokenUsage:
        """Get token usage statistics."""
        return self.token_usage
    
    @abstractmethod
    def supports_reasoning_budget(self) -> bool:
        """Check if this model supports reasoning budget features."""
        pass
    
    def update_token_usage(self, prompt_tokens: int, completion_tokens: int):
        """Update token usage statistics."""
        self.token_usage.prompt_tokens += prompt_tokens
        self.token_usage.completion_tokens += completion_tokens
        self.token_usage.total_tokens += (prompt_tokens + completion_tokens)
    
    def reset_token_usage(self):
        """Reset token usage statistics."""
        self.token_usage = TokenUsage()
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model-specific information and capabilities."""
        pass 