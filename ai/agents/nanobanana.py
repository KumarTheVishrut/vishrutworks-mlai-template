from agents.base import BaseAgent, ProviderError
from typing import AsyncGenerator

class NanobananaAgent(BaseAgent):
    provider_name = "nanobanana"

    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        # Mock image generation response
        try:
            # Simulate a URL being yielded
            image_url = f"https://mock-image-api.com/generate?prompt={prompt}"
            yield f"Generated image: {image_url}"
        except Exception as e:
            raise ProviderError(f"Nanobanana agent error: {str(e)}")
