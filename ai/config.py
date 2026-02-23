import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict

class Settings(BaseSettings):
    # Provider API Keys
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379")

    # Security
    INTERNAL_API_SECRET: str = os.getenv("INTERNAL_API_SECRET", "change-me")

    # Model Config
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "claude")             # claude | gemini | openai
    
    PROVIDER_TIMEOUTS: Dict[str, int] = {
        "claude": int(os.getenv("CLAUDE_TIMEOUT_SECONDS", 30)),
        "gemini": int(os.getenv("GEMINI_TIMEOUT_SECONDS", 20)),
        "openai": int(os.getenv("OPENAI_TIMEOUT_SECONDS", 30)),
    }

    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", 300))

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
