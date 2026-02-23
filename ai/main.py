from fastapi import FastAPI, Header, HTTPException, Depends, Response
from fastapi.responses import StreamingResponse
from schemas import ChatRequest
from router import ModelRouter
from config import settings
from cost.token_counter import count_tokens_anthropic, count_tokens_openai
import hashlib

app = FastAPI(title="AI Inference Service")
router = ModelRouter()

async def verify_internal_secret(x_internal_secret: str = Header(None)):
    if x_internal_secret != settings.INTERNAL_API_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized")

@app.post("/ai/chat")
async def chat(
    request: ChatRequest,
    x_internal_secret: str = Depends(verify_internal_secret)
):
    # Calculate input tokens before starting the stream
    # We'll use the default model for estimation if not specified
    provider_for_tokens = settings.DEFAULT_MODEL
    if provider_for_tokens == "claude":
        token_count = count_tokens_anthropic(request.prompt)
    else:
        token_count = count_tokens_openai(request.prompt)

    async def event_generator():
        try:
            async for token, provider in router.route(request.prompt, request.local_model_id, request.task_type):
                yield token
        except Exception as e:
            yield f"Error: {str(e)}"

    headers = {
        "X-Tokens-Used": str(token_count),
        "X-Provider-Used": settings.DEFAULT_MODEL, 
    }

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
