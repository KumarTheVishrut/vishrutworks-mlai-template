from abc import ABC, abstractmethod
from typing import AsyncGenerator

class ProviderError(Exception):
    """Raised when a provider returns an error response."""
    pass

class BaseAgent(ABC):
    provider_name: str  # Must be set on the class, not instance

    @abstractmethod
    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Yields individual text tokens as plain str.
        
        Rules:
        - Never yield dicts, JSON, or structured objects
        - Raise ProviderError on any API failure
        - Raise TimeoutError if provider doesn't respond within timeout
        - Never swallow exceptions silently
        """
        ...
