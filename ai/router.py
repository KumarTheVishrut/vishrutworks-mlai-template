import asyncio
import hashlib
import logging
from typing import AsyncGenerator, Type
import anthropic
import openai
from agents.base import BaseAgent, ProviderError
from agents.claude import ClaudeAgent
from agents.gemini import GeminiAgent
from agents.openai import OpenAIAgent
from agents.nanobanana import NanobananaAgent
from cache.redis_cache import get_cached, set_cached
from local.local_model import load_local_model
from config import settings

logger = logging.getLogger(__name__)

# Provider priority order
PROVIDER_CHAIN: list[Type[BaseAgent]] = [ClaudeAgent, GeminiAgent, OpenAIAgent]

class AllProvidersFailedError(Exception):
    """Raised when all configured LLM providers fail."""
    pass

class ModelRouter:
    def __init__(self):
        self.agents = {cls.provider_name: cls() for cls in PROVIDER_CHAIN}
        self.agents["nanobanana"] = NanobananaAgent()

    def _generate_cache_key(self, prompt: str, local_model_id: str | None) -> str:
        model_part = local_model_id if local_model_id else "cloud"
        combined = f"{prompt}:{model_part}"
        return f"prompt:{hashlib.sha256(combined.encode()).hexdigest()}"

    async def route(self, prompt: str, local_model_id: str | None = None, task_type: str = "chat") -> AsyncGenerator[tuple[str, str], None]:
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        
        # Step 0: Image task handling
        if task_type == "image":
            async for token in self.agents["nanobanana"].stream(prompt):
                yield token, "nanobanana"
            return

        # Step 1: Check local weights first
        if local_model_id:
            local = load_local_model(local_model_id)
            if local:
                logger.info(f"Using local model: {local_model_id} for prompt hash {prompt_hash}")
                async for token in local.stream(prompt):
                    yield token, "local"
                return

        # Step 2: Check Redis cache
        cache_key = self._generate_cache_key(prompt, local_model_id)
        cached = await get_cached(cache_key)
        if cached:
            logger.info(f"Cache hit for prompt hash {prompt_hash}")
            yield cached, "cache"
            return

        # Step 3: Try providers in order (Circuit Breaker)
        for AgentClass in PROVIDER_CHAIN:
            provider_name = AgentClass.provider_name
            agent = self.agents[provider_name]
            timeout = settings.PROVIDER_TIMEOUTS.get(provider_name, 30)
            
            full_response = ""
            try:
                logger.info(f"Trying provider: {provider_name} for prompt hash {prompt_hash}")
                
                try:
                    # We wrap the generator to handle timeout for individual tokens
                    gen = agent.stream(prompt)
                    while True:
                        try:
                            # Use wait_for for each token to avoid hanging indefinitely
                            # although the timeout is usually for the entire response
                            token = await asyncio.wait_for(anext(gen), timeout=timeout)
                            yield token, provider_name
                            full_response += token
                        except StopAsyncIteration:
                            break
                    
                    # Cache successful result ONLY on full success
                    if full_response:
                        await set_cached(cache_key, full_response, ttl=settings.CACHE_TTL_SECONDS)
                    return # SUCCESS: exit loop
                
                except (asyncio.TimeoutError, ProviderError, anthropic.APIError, openai.APIError) as e:
                    logger.warning(f"[CIRCUIT BREAKER] {provider_name} failed: {type(e).__name__} for prompt hash {prompt_hash}")
                    full_response = "" # Reset for next provider
                    continue # Try next provider
                
            except Exception as e:
                logger.error(f"Unexpected error with provider {provider_name}: {e} for prompt hash {prompt_hash}")
                full_response = ""
                continue
        
        raise AllProvidersFailedError("All LLM providers are unavailable.")
