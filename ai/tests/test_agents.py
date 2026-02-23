import pytest
from agents.claude import ClaudeAgent
from agents.gemini import GeminiAgent
from agents.openai import OpenAIAgent
from unittest.mock import AsyncMock, patch, MagicMock

async def async_gen(items):
    for item in items:
        yield item

@pytest.mark.asyncio
async def test_claude_agent_stream():
    with patch("anthropic.AsyncAnthropic") as mock_anthropic:
        mock_client = mock_anthropic.return_value
        mock_stream = AsyncMock()
        mock_stream.__aenter__.return_value.text_stream = async_gen(["Hello", " world"])
        mock_client.messages.stream.return_value = mock_stream
        
        agent = ClaudeAgent()
        tokens = []
        async for token in agent.stream("test prompt"):
            tokens.append(token)
        
        assert tokens == ["Hello", " world"]

@pytest.mark.asyncio
async def test_gemini_agent_stream():
    with patch("google.generativeai.GenerativeModel") as mock_model_class:
        mock_model = mock_model_class.return_value
        
        # In Python 3.8+, AsyncMock handles __aiter__ if return_value is an iterable
        mock_response = AsyncMock()
        mock_response.__aiter__.return_value = [type('obj', (object,), {'text': x})() for x in ["Hello", " world"]]
        
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        
        agent = GeminiAgent()
        tokens = []
        async for token in agent.stream("test prompt"):
            tokens.append(token)
        
        assert tokens == ["Hello", " world"]

@pytest.mark.asyncio
async def test_openai_agent_stream():
    with patch("openai.AsyncOpenAI") as mock_openai:
        mock_client = mock_openai.return_value
        
        mock_stream = AsyncMock()
        mock_stream.__aiter__.return_value = [
            type('obj', (object,), {'choices': [type('obj', (object,), {'delta': type('obj', (object,), {'content': 'Hello'})()})()]})(),
            type('obj', (object,), {'choices': [type('obj', (object,), {'delta': type('obj', (object,), {'content': ' world'})()})()]})(),
        ]
        
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)
        
        agent = OpenAIAgent()
        tokens = []
        async for token in agent.stream("test prompt"):
            tokens.append(token)
        
        assert tokens == ["Hello", " world"]
