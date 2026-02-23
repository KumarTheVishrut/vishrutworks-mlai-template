import anthropic
from agents.base import BaseAgent, ProviderError
from typing import AsyncGenerator
from config import settings

class ClaudeAgent(BaseAgent):
    provider_name = "claude"

    def __init__(self):
        self.client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        try:
            async with self.client.messages.stream(
                model="claude-3-5-sonnet-20240620",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            raise ProviderError(f"Claude agent error: {str(e)}")
