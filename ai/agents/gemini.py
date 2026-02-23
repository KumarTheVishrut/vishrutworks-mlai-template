import google.generativeai as genai
from agents.base import BaseAgent, ProviderError
from typing import AsyncGenerator
from config import settings

class GeminiAgent(BaseAgent):
    provider_name = "gemini"

    def __init__(self):
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        try:
            response = await self.model.generate_content_async(prompt, stream=True)
            async for chunk in response:
                yield chunk.text
        except Exception as e:
            raise ProviderError(f"Gemini agent error: {str(e)}")
