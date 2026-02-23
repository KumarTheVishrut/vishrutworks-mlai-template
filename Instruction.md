# 🤖 AI Inference Agent Instructions — Multi-LLM Model Router

> **Repo:** `ai/`  
> **Purpose:** Directives and execution logic for AI coding agents working on this codebase.  
> **Last Updated:** 2026-02-23

---

## ⚠️ Global Constraints

| Constraint | Rule |
|---|---|
| Agent Interface | All agents must extend `BaseAgent` and implement `stream()` |
| Output Format | `stream()` must yield `str` tokens **only** — no dicts, no metadata |
| Failover Order | Claude → Gemini → OpenAI. Never change this order without explicit instruction. |
| Cache First | Always check Redis before calling any cloud provider |
| Local Model First | Always check `./weights/` before Redis if `local_model_id` is set |
| Cost Guard | Always return `X-Tokens-Used` in response headers |
| Internal Auth | All endpoints must validate `X-Internal-Secret` header |

---

## Global State Reference

```
Service State
├── ModelRouter (singleton)
│   ├── agents: { claude, gemini, openai, nanobanana }
│   └── route(prompt, local_model_id) → AsyncGenerator[str]
│
├── Redis Cache
│   ├── TTL: 300 seconds
│   ├── Key: prompt:{sha256(prompt + model_id)}
│   └── Eviction: allkeys-lru
│
├── Local Weights
│   └── ./weights/<model_id>/   → HuggingFace pipeline
│
└── Provider Chain (priority order)
    1. ClaudeAgent  (Default — best quality)
    2. GeminiAgent  (Failover — low latency)
    3. OpenAIAgent  (Secondary failover)
```

---

## Step-by-Step Execution Logic

### Task 1 — Add a New Provider Agent

```
1. Create agents/<provider>.py
2. Import and extend BaseAgent
3. Set class attribute: provider_name = "<provider>"
4. Implement async def stream(self, prompt: str) -> AsyncGenerator[str, None]
5. Wrap provider SDK call in try/except — raise ProviderError on failure
6. Add timeout using asyncio.wait_for(agent.stream(prompt), timeout=TIMEOUT)
7. Register in PROVIDER_CHAIN in router.py at the correct priority position
8. Add provider API key to .env.example and config.py
9. Write unit test in tests/test_agents.py using mock API responses
```

### Task 2 — Modify Failover Logic

```
1. Open router.py → ModelRouter.route()
2. PROVIDER_CHAIN list defines priority — reorder only if explicitly instructed
3. Caught exception types: TimeoutError, ProviderError, anthropic.APIError, openai.APIError
4. Each failure must log: provider name, error type, prompt hash (not prompt text)
5. Reset full_response = "" between provider attempts
6. Only cache on SUCCESS — never cache partial or empty responses
```

### Task 3 — Implement Caching

```
1. Generate cache key: sha256(f"{prompt}:{model_id}").hexdigest()
2. Check cache BEFORE local model check and BEFORE cloud provider calls
3. If cache hit: yield the cached string and return immediately
4. If cache miss: stream provider, accumulate full_response, store in Redis after stream ends
5. TTL = 300 seconds (configurable via env var CACHE_TTL_SECONDS)
```

### Task 4 — Add Local Model Support

```
1. Check ./weights/<local_model_id>/ existence with pathlib.Path
2. If path exists: load HuggingFace pipeline, stream output, return early
3. If path does not exist: fall through to Redis check → cloud provider chain
4. Local model output must also go through the unified str token generator
5. Never call cloud APIs if a valid local model is found
```

---

## Directive: BaseAgent Contract

Every provider agent **must** follow this exact interface:

```python
# agents/base.py — DO NOT MODIFY THIS FILE

from abc import ABC, abstractmethod
from typing import AsyncGenerator

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

class ProviderError(Exception):
    """Raised when a provider returns an error response."""
    pass
```

---

## Directive: Streaming Standardization

The backend receives a unified SSE stream. Every provider must map its SDK output to plain text tokens:

```python
# ✅ CORRECT — yields str tokens
async def stream(self, prompt: str):
    async for chunk in sdk_call():
        yield chunk.text  # str

# ❌ WRONG — yields structured data
async def stream(self, prompt: str):
    async for chunk in sdk_call():
        yield {"token": chunk.text, "model": "claude"}  # dict — breaks the router
```

---

## Directive: Circuit Breaker Implementation

```python
# ✅ CORRECT circuit breaker pattern
for AgentClass in PROVIDER_CHAIN:
    try:
        async for token in asyncio.wait_for(
            agent.stream(prompt),
            timeout=settings.PROVIDER_TIMEOUTS[AgentClass.provider_name]
        ):
            yield token
        return  # ← SUCCESS: exit loop
    
    except (asyncio.TimeoutError, ProviderError) as e:
        logger.warning(f"[CIRCUIT BREAKER] {AgentClass.provider_name} failed: {type(e).__name__}")
        continue  # ← Try next provider

raise AllProvidersFailedError()

# ❌ WRONG — bare except hides bugs
try:
    ...
except Exception:
    pass
```

---

## Directive: Local Model Priority

```python
# Decision tree — implement in this exact order
async def route(self, prompt: str, local_model_id: str | None = None):
    
    # 1. LOCAL MODEL — highest priority
    if local_model_id:
        model = load_local_model(local_model_id)
        if model:
            # stream and return — skip all cloud providers
            ...
            return
    
    # 2. REDIS CACHE — before any cloud call
    cached = await get_cached(cache_key)
    if cached:
        yield cached
        return
    
    # 3. CLOUD PROVIDER CHAIN — only if no cache hit
    for provider in PROVIDER_CHAIN:
        ...
```

---

## Directive: Cost Guard

Token count must be calculated and returned with every non-cached response:

```python
# In the route() method, after streaming completes:
if provider_name == "claude":
    token_count = count_tokens_anthropic(full_response)
elif provider_name in ("openai",):
    token_count = count_tokens_openai(full_response)

# Return in response header (set in the FastAPI endpoint wrapper):
response.headers["X-Tokens-Used"] = str(token_count)
response.headers["X-Provider-Used"] = provider_name
```

---

## Directive: Internal Auth

All endpoints are internal-only (called only by the backend). Validate the shared secret:

```python
from fastapi import Header, HTTPException

async def verify_internal_secret(
    x_internal_secret: str = Header(...),
):
    if x_internal_secret != settings.INTERNAL_API_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized")
```

Add as a dependency on every router endpoint.

---

## Anti-Patterns — Never Do These

```
❌ Never yield dicts or structured objects from stream() — str tokens only
❌ Never change PROVIDER_CHAIN order without explicit instruction
❌ Never cache empty or partial responses
❌ Never call cloud APIs without first checking local weights (if local_model_id set)
❌ Never use bare except — always catch specific exception types
❌ Never log the raw prompt text — only log the prompt hash for privacy
❌ Never call multiple providers simultaneously — sequential failover only
❌ Never import provider SDKs at module level without try/except for missing keys
```

---

## Checklist Before Marking a Task Complete

- [ ] New agent extends `BaseAgent` and implements `stream()` returning `AsyncGenerator[str, None]`
- [ ] `provider_name` class attribute is set
- [ ] Agent is registered in `PROVIDER_CHAIN` in `router.py`
- [ ] Timeout is configured in `settings.PROVIDER_TIMEOUTS`
- [ ] API key added to `.env.example` and `config.py`
- [ ] `X-Tokens-Used` and `X-Provider-Used` headers returned
- [ ] Unit tests added with mocked SDK responses
- [ ] Cache key uses sha256 hash, not raw prompt text

---

> **See Also:** [ARCHITECTURE.md](./ARCHITECTURE.md) · [Backend INSTRUCTION.md](../backend/INSTRUCTION.md)
