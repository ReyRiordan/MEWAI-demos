"""
Configuration settings for the standalone patient simulation API.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # AI services
    openrouter_api_key: str = Field(..., description="OpenRouter API key")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter base URL",
    )
    fireworks_api_key: str = Field(..., description="Fireworks API key for Whisper STT")
    fireworks_base_url: str = Field(
        default="https://audio-prod.api.fireworks.ai/v1/audio/transcriptions",
        description="Fireworks STT endpoint",
    )
    inworld_api_key: Optional[str] = Field(default=None, description="Inworld TTS API key")

    # Twilio TURN credentials (optional)
    twilio_account_sid: Optional[str] = Field(default=None)
    twilio_auth_token: Optional[str] = Field(default=None)

    # Simulation defaults
    concurrency_limit: int = Field(default=5)

    # CORS
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        description="Comma-separated list of allowed CORS origins",
    )

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


settings = Settings()
