import tiktoken
import anthropic
from config import settings

def count_tokens_openai(text: str, model: str = "gpt-4o") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        # Fallback to simple whitespace splitting if model not found or error
        return len(text.split())

def count_tokens_anthropic(text: str) -> int:
    try:
        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        response = client.beta.messages.count_tokens(
            model="claude-3-5-sonnet-20240620",
            messages=[{"role": "user", "content": text}],
        )
        return response.input_tokens
    except Exception:
        # Fallback to simple whitespace splitting
        return len(text.split())
