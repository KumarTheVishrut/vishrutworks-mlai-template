# saas-ai

> Multi-LLM model router for a B2B AI SaaS — provider failover, Redis caching, streaming standardization, and local model support.

![FastAPI](https://img.shields.io/badge/FastAPI-latest-009688?logo=fastapi)
![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Claude](https://img.shields.io/badge/Default_Model-Claude_3.5_Sonnet-orange)
![Redis](https://img.shields.io/badge/Cache-Redis-red?logo=redis)

---

## What this is

`saas-ai` is a **stateless model router** that sits between the backend and LLM providers. It abstracts provider differences behind a single streaming interface and handles:

- **Provider failover** — if Claude times out, automatically retries with Gemini, then OpenAI
- **Prompt caching** — identical prompts within 5 minutes return cached results from Redis
- **Local model support** — checks `./weights/` before calling any cloud provider
- **Cost tracking** — returns token count in response headers so the backend can deduct credits

The backend calls this service. Nothing else does.

---

## Tech stack

| Concern | Tool |
|---|---|
| Framework | FastAPI |
| Default model | Claude Sonnet (Anthropic SDK) |
| Failover #1 | Gemini 1.5 Flash (Google Generative AI) |
| Failover #2 | GPT-4o / O1 (OpenAI SDK) |
| Image/visual | nanobanana |
| Cache | Redis (5-min TTL) |
| Local ML | HuggingFace Transformers |
| Token counting | `tiktoken` (OpenAI) + Anthropic SDK |

---

## Prerequisites

- Python 3.11+
- Redis (or Docker — see [saas-infra](../infra))
- API keys for at least one LLM provider (Claude recommended as primary)

---

## Getting started

```bash
# 1. Clone
git clone https://github.com/your-org/saas-ai.git
cd saas-ai

# 2. Create virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Fill in values — see Environment Variables below

# 5. Start dev server
uvicorn main:app --reload --port 8001
```

---

## Environment variables

```bash
# .env

# LLM Provider Keys — configure at least one
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AI...

# Redis
REDIS_URL=redis://localhost:6379

# Internal auth — must match saas-backend's INTERNAL_API_SECRET
INTERNAL_API_SECRET=...

# Model config
DEFAULT_MODEL=claude
CLAUDE_TIMEOUT_SECONDS=30
GEMINI_TIMEOUT_SECONDS=20
OPENAI_TIMEOUT_SECONDS=30
CACHE_TTL_SECONDS=300
```

---

## Project structure

```
agents/
├── base.py          # Abstract BaseAgent — all providers implement this
├── claude.py        # Claude Sonnet (default)
├── openai.py        # GPT-4o / O1
├── gemini.py        # Gemini 1.5 Flash
└── nanobanana.py    # Image generation

cache/
└── redis_cache.py   # Redis get/set with TTL

local/
└── local_model.py   # HuggingFace pipeline loader from ./weights/

cost/
└── token_counter.py # tiktoken + Anthropic SDK token estimation

weights/             # Local model weight directories (git-ignored)
router.py            # ModelRouter — orchestrates the full call chain
main.py              # FastAPI app + /ai/chat endpoint
schemas.py           # Pydantic V2 request/response schemas
```

---

## How the router works

Every incoming request goes through this decision chain:

```
Request arrives
      │
      ▼
1. local_model_id set? → load from ./weights/ → stream → done
      │ no
      ▼
2. Prompt in Redis cache? → return cached result → done
      │ no
      ▼
3. Try Claude  ──── timeout/error ──→ Try Gemini  ──── timeout/error ──→ Try OpenAI
      │ success                            │ success                          │ success
      ▼                                    ▼                                  ▼
   Cache result                        Cache result                       Cache result
   Return stream                       Return stream                      Return stream
```

---

## Provider failover chain

| Priority | Provider | Model | Trigger |
|---|---|---|---|
| 1 (default) | Anthropic | Claude Sonnet | All requests |
| 2 (failover) | Google | Gemini 1.5 Flash | Claude timeout or error |
| 3 (fallback) | OpenAI | GPT-4o | Gemini timeout or error |

The order is defined in `PROVIDER_CHAIN` in `router.py`.

---

## Adding a local model

Place model weights in `./weights/<model_id>/` as a HuggingFace-compatible directory, then pass `local_model_id` in the request body. The router will load the local pipeline and skip all cloud providers.

```bash
# Example: add a fine-tuned model
cp -r /path/to/my-model ./weights/my-model-v1

# Test it
curl -X POST http://localhost:8001/ai/chat \
  -H "X-Internal-Secret: $INTERNAL_API_SECRET" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "local_model_id": "my-model-v1"}'
```

---

## Response headers

Every response includes:

| Header | Value | Description |
|---|---|---|
| `X-Tokens-Used` | integer | Estimated token count for credit deduction |
| `X-Provider-Used` | string | Which provider actually served the request |

---

## Key conventions

**All agents extend `BaseAgent`** — `stream()` must yield plain `str` tokens only. No dicts, no metadata in the stream. This is what keeps the backend provider-agnostic.

**This service is internal-only** — every endpoint validates the `X-Internal-Secret` header. It is not exposed publicly via Nginx.

**Cache on success only** — partial or empty responses are never written to Redis.

---

## Scripts

```bash
uvicorn main:app --reload --port 8001   # Dev server
pytest tests/ -v                        # Run tests
```

---

## Related repos

| Repo | Role |
|---|---|
| [saas-frontend](../frontend) | Next.js dashboard |
| [saas-backend](../backend) | API — calls this service |
| [saas-infra](../infra) | Docker Compose, Nginx, deployment |

---

## Docs

- [ARCHITECTURE.md](./ARCHITECTURE.md) — circuit breaker design, streaming standardization, Redis caching, cost guard
- [INSTRUCTION.md](./INSTRUCTION.md) — agent directives, BaseAgent contract, failover rules, anti-patterns
