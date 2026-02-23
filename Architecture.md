# 🤖 AI Inference Architecture — Multi-LLM Model Router

> **Repo:** `ai/`  
> **Stack:** Python · FastAPI · OpenAI SDK · Anthropic SDK · Google Gemini SDK · Redis · HuggingFace Transformers

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Technology Stack](#technology-stack)
3. [Directory Structure](#directory-structure)
4. [Model Router](#model-router)
5. [Circuit Breaker & Failover](#circuit-breaker--failover)
6. [Streaming Standardization](#streaming-standardization)
7. [Provider Agents](#provider-agents)
8. [Redis Caching](#redis-caching)
9. [Local Model Support](#local-model-support)
10. [Cost Guard](#cost-guard)
11. [Environment Variables](#environment-variables)
12. [Scripts & Commands](#scripts--commands)

---

## System Overview

The AI Inference service is a **stateless Model Router** designed for maximum uptime via provider-failover logic. It abstracts all LLM providers behind a unified interface so the backend never needs to know which model served a request.

```
                  ┌─────────────────────────────────────────┐
                  │           AI Inference Service           │
                  │                                         │
  Backend ───────►│  POST /ai/chat                          │
                  │         │                               │
                  │   ┌─────▼──────────────────┐            │
                  │   │      ModelRouter        │            │
                  │   │                        │            │
                  │   │  1. Check Redis cache  │            │
                  │   │  2. Check /weights/    │            │
                  │   │  3. Route to provider  │            │
                  │   └─────┬──────────────────┘            │
                  │         │                               │
                  │   ┌─────▼─────────────────────────┐     │
                  │   │      Provider Selection         │     │
                  │   │                               │     │
                  │   │  Claude (Default B2B)         │     │
                  │   │    ↓ on timeout               │     │
                  │   │  Gemini Flash (Failover)      │     │
                  │   │    ↓ on timeout               │     │
                  │   │  GPT-4o (Secondary Failover)  │     │
                  │   └───────────────────────────────┘     │
                  │                                         │
                  └─────────────────────────────────────────┘
```

---

## Technology Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| Framework | FastAPI | Latest | Async API serving |
| Default Model | Claude Sonnet (Anthropic) | 3.5 | B2B default — best reasoning |
| Failover Model | Gemini 1.5 Flash (Google) | Latest | Low-latency fallback |
| Secondary | GPT-4o / O1 (OpenAI) | Latest | Secondary failover + specialized tasks |
| Image Gen | nanobanana | Custom | Image generation & text-to-visual |
| Caching | Redis | Latest | 5-min prompt result cache |
| Local ML | HuggingFace Transformers | Latest | On-prem weights for niche B2B tasks |
| Token Counting | `tiktoken` (OpenAI) + Anthropic SDK | Latest | Cost estimation |

---

## Directory Structure

```
ai/
├── main.py                       # FastAPI app, registers /ai/chat endpoint
├── router.py                     # ModelRouter class — core orchestration logic
│
├── agents/
│   ├── base.py                   # Abstract base class: BaseAgent
│   ├── claude.py                 # Claude 3.5 Sonnet agent (DEFAULT)
│   ├── openai.py                 # GPT-4o / O1 agent
│   ├── gemini.py                 # Gemini 1.5 Flash agent
│   └── nanobanana.py             # Image generation & text-to-visual agent
│
├── cache/
│   └── redis_cache.py            # Redis get/set with 5-min TTL
│
├── local/
│   └── local_model.py            # HuggingFace pipeline loader from /weights/
│
├── cost/
│   └── token_counter.py          # tiktoken + anthropic-sdk token estimation
│
├── weights/                      # Local model weight files (not git-tracked)
│   └── .gitkeep
│
├── schemas.py                    # Pydantic V2 request/response schemas
├── config.py                     # Settings
├── .env.example
├── requirements.txt
└── Dockerfile
```

---

## Model Router

The `ModelRouter` class is the heart of the service. It implements provider selection, caching, local model priority, and failover.

```python
# router.py
from agents.claude import ClaudeAgent
from agents.gemini import GeminiAgent
from agents.openai import OpenAIAgent
from cache.redis_cache import get_cached, set_cached
from local.local_model import load_local_model
from cost.token_counter import count_tokens
import asyncio

# Provider priority order
PROVIDER_CHAIN = [ClaudeAgent, GeminiAgent, OpenAIAgent]

class ModelRouter:
    def __init__(self):
        self.agents = {cls.provider_name: cls() for cls in PROVIDER_CHAIN}

    async def route(self, prompt: str, local_model_id: str | None = None) -> AsyncGenerator[str, None]:
        
        # Step 1: Check local weights first
        if local_model_id:
            local = load_local_model(local_model_id)
            if local:
                async for token in local.stream(prompt):
                    yield token
                return

        # Step 2: Check Redis cache
        cache_key = f"prompt:{hash(prompt)}"
        cached = await get_cached(cache_key)
        if cached:
            yield cached
            return

        # Step 3: Try providers in order (Circuit Breaker)
        full_response = ""
        for AgentClass in PROVIDER_CHAIN:
            agent = self.agents[AgentClass.provider_name]
            try:
                async for token in agent.stream(prompt):
                    full_response += token
                    yield token
                
                # Cache successful result
                await set_cached(cache_key, full_response, ttl=300)
                return
                
            except (TimeoutError, ProviderError) as e:
                logger.warning(f"Provider {AgentClass.provider_name} failed: {e}. Trying next...")
                full_response = ""
                continue
        
        raise AllProvidersFailedError("All LLM providers are unavailable.")
```

---

## Circuit Breaker & Failover

The failover chain is: **Claude → Gemini → OpenAI**

```
Attempt Claude
    │
    ├── Success → return stream, cache result
    │
    └── Timeout/Error
            │
        Attempt Gemini
            │
            ├── Success → return stream, cache result
            │
            └── Timeout/Error
                    │
                Attempt OpenAI
                    │
                    ├── Success → return stream, cache result
                    │
                    └── All failed → raise AllProvidersFailedError
```

**Timeout Configuration:**

| Provider | Timeout |
|---|---|
| Claude | 30 seconds |
| Gemini | 20 seconds |
| OpenAI | 30 seconds |

---

## Streaming Standardization

All providers expose different streaming APIs. The `BaseAgent` class normalizes them into a single `AsyncGenerator[str, None]`.

```python
# agents/base.py
from abc import ABC, abstractmethod
from typing import AsyncGenerator

class BaseAgent(ABC):
    provider_name: str

    @abstractmethod
    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Must yield individual text tokens as strings.
        The ModelRouter expects a consistent str stream — no metadata, no dicts.
        """
        ...
```

```python
# agents/claude.py
import anthropic
from agents.base import BaseAgent

class ClaudeAgent(BaseAgent):
    provider_name = "claude"

    def __init__(self):
        self.client = anthropic.AsyncAnthropic()

    async def stream(self, prompt: str):
        async with self.client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text
```

```python
# agents/gemini.py
import google.generativeai as genai
from agents.base import BaseAgent

class GeminiAgent(BaseAgent):
    provider_name = "gemini"

    async def stream(self, prompt: str):
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = await model.generate_content_async(prompt, stream=True)
        async for chunk in response:
            yield chunk.text
```

---

## Provider Agents

### Claude (Default B2B)
- **Model:** `claude-sonnet-4-6`
- **Use case:** All standard B2B tasks — analysis, generation, summarization
- **SDK:** `anthropic` Python SDK

### GPT-4o / O1 (OpenAI)
- **Models:** `gpt-4o`, `o1-preview`
- **Use case:** Secondary failover; coding tasks; when OpenAI tool-use is needed
- **SDK:** `openai` Python SDK

### Gemini 1.5 Flash (Google)
- **Model:** `gemini-1.5-flash`
- **Use case:** Primary failover; low-latency tasks; cost-sensitive workloads
- **SDK:** `google-generativeai`

### nanobanana (Image/Visual)
- **Use case:** Image generation and text-to-visual tasks
- **Triggered:** When `task_type="image"` is passed in the request

---

## Redis Caching

Identical prompts within a 5-minute window return cached results instantly — saving API costs significantly for repeated B2B queries.

```python
# cache/redis_cache.py
import redis.asyncio as redis
import json

r = redis.from_url("redis://redis:6379")

async def get_cached(key: str) -> str | None:
    value = await r.get(key)
    return value.decode() if value else None

async def set_cached(key: str, value: str, ttl: int = 300):
    await r.setex(key, ttl, value.encode())
```

**Cache Key Strategy:** `prompt:{sha256_hash(prompt + model_id)}`  
**TTL:** 300 seconds (5 minutes)  
**Eviction Policy:** `allkeys-lru`

---

## Local Model Support

If a `local_model_id` is passed in the request, the router searches `./weights/` **before** calling any cloud provider.

```python
# local/local_model.py
from pathlib import Path
from transformers import pipeline

WEIGHTS_DIR = Path("./weights")

def load_local_model(model_id: str):
    """Returns a HuggingFace pipeline if weights exist, else None."""
    model_path = WEIGHTS_DIR / model_id
    if not model_path.exists():
        return None
    
    return pipeline("text-generation", model=str(model_path))
```

Place model weight directories in `./weights/<model_id>/` to enable local inference.

---

## Cost Guard

Token counts are estimated **before** returning the response to enable credit deduction upstream.

```python
# cost/token_counter.py
import tiktoken
import anthropic

def count_tokens_openai(text: str, model: str = "gpt-4o") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def count_tokens_anthropic(text: str) -> int:
    client = anthropic.Anthropic()
    response = client.beta.messages.count_tokens(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": text}],
    )
    return response.input_tokens
```

The token count is returned in the response headers (`X-Tokens-Used`) so the backend can deduct credits accurately.

---

## Environment Variables

```bash
# .env.example

# Provider API Keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AI...

# Redis
REDIS_URL=redis://redis:6379

# Security
INTERNAL_API_SECRET=...        # Shared with Backend — validates internal calls

# Model Config
DEFAULT_MODEL=claude             # claude | gemini | openai
CLAUDE_TIMEOUT_SECONDS=30
GEMINI_TIMEOUT_SECONDS=20
OPENAI_TIMEOUT_SECONDS=30
```

---

## Scripts & Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run service
uvicorn main:app --reload --port 8001

# Test failover manually
curl -X POST http://localhost:8001/ai/chat \
  -H "Content-Type: application/json" \
  -H "X-Internal-Secret: $INTERNAL_API_SECRET" \
  -d '{"prompt": "Hello world", "local_model_id": null}'

# Load test
locust -f tests/locustfile.py --headless -u 50 -r 5
```

---

> **Related Docs:** [Backend ARCHITECTURE.md](../backend/ARCHITECTURE.md) · [AI INSTRUCTION.md](./INSTRUCTION.md) · [Infra ARCHITECTURE.md](../infra/ARCHITECTURE.md)
