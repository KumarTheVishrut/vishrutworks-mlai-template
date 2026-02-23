import pytest
from router import ModelRouter
from unittest.mock import AsyncMock, patch
from agents.claude import ClaudeAgent
from agents.gemini import GeminiAgent

async def async_gen(items):
    for item in items:
        yield item

@pytest.mark.asyncio
async def test_router_success_first_provider():
    router = ModelRouter()
    
    with patch.object(ClaudeAgent, "stream", return_value=async_gen(["Hello", " world"])):
        with patch("router.get_cached", return_value=None):
            with patch("router.set_cached", return_value=None):
                tokens = []
                async for token, provider in router.route("test prompt"):
                    tokens.append(token)
                    assert provider == "claude"
                
                assert tokens == ["Hello", " world"]

@pytest.mark.asyncio
async def test_router_failover():
    router = ModelRouter()
    
    # Claude fails, Gemini succeeds
    async def failing_gen(prompt):
        if False: yield # make it a generator
        raise Exception("Claude failed")
    
    with patch.object(ClaudeAgent, "stream", side_effect=failing_gen):
        with patch.object(GeminiAgent, "stream", return_value=async_gen(["Hi", " from", " Gemini"])):
            with patch("router.get_cached", return_value=None):
                with patch("router.set_cached", return_value=None):
                    tokens = []
                    async for token, provider in router.route("test prompt"):
                        tokens.append(token)
                        assert provider == "gemini"
                    
                    assert tokens == ["Hi", " from", " Gemini"]
