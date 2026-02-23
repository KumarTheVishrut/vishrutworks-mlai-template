from pathlib import Path
from transformers import pipeline
import asyncio
from typing import AsyncGenerator

WEIGHTS_DIR = Path("./weights")

class LocalModel:
    def __init__(self, model_id: str):
        self.model_path = WEIGHTS_DIR / model_id
        self.generator = pipeline("text-generation", model=str(self.model_path))

    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        # Simple simulation of streaming for local models
        result = self.generator(prompt, max_new_tokens=2048)
        generated_text = result[0]['generated_text']
        # Return only the new part of the text
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]
        
        # Simulate streaming by yielding word by word
        for word in generated_text.split(" "):
            yield word + " "
            await asyncio.sleep(0.01)

def load_local_model(model_id: str):
    """Returns a LocalModel instance if weights exist, else None."""
    model_path = WEIGHTS_DIR / model_id
    if not model_path.exists() or not model_path.is_dir():
        return None
    
    return LocalModel(model_id)
