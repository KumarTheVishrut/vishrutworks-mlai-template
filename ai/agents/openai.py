import openai
from agents.base import BaseAgent, ProviderError
from typing import AsyncGenerator
from config import settings

class OpenAIAgent(BaseAgent):
    provider_name = "openai"

    def __init__(self):
        self.client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        try:
            stream = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise ProviderError(f"OpenAI agent error: {str(e)}")
