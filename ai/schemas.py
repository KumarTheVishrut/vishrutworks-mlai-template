from pydantic import BaseModel, Field
from typing import Optional, List

class ChatRequest(BaseModel):
    prompt: str = Field(..., description="The prompt to generate a response from.")
    local_model_id: Optional[str] = Field(None, description="The ID of the local model to use, if any.")
    task_type: Optional[str] = Field("chat", description="The type of task, e.g., 'chat', 'image'.")

class ChatResponse(BaseModel):
    text: str = Field(..., description="The generated text response.")
    provider: str = Field(..., description="The name of the provider used.")
    tokens_used: int = Field(..., description="The number of tokens used.")
