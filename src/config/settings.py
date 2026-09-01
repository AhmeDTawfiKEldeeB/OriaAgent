from enum import Enum
from functools import lru_cache
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProviderType(str, Enum):
    GEMINI = "gemini"
    GROQ = "groq"


class GeminiSettings(BaseModel):
    """Configuration settings for Google Gemini provider."""
    api_key: str = Field(default="", description="Google Gemini API Key")
    model: str = Field(default="gemini-2.0-flash", description="Gemini Model Name")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_output_tokens: Optional[int] = Field(default=None, description="Max output tokens")


class GroqSettings(BaseModel):
    """Configuration settings for Groq provider."""
    api_key: str = Field(default="", description="Groq API Key")
    model: str = Field(default="llama-3.3-70b-versatile", description="Groq Model Name")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Max completion tokens")


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Active LLM provider
    llm_provider: LLMProviderType = Field(
        default=LLMProviderType.GEMINI,
        validation_alias="LLM_PROVIDER",
        description="The active LLM provider (gemini, groq)",
    )

    # Environment
    app_env: str = Field(default="development", validation_alias="APP_ENV")

    # Gemini Config
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.0-flash", validation_alias="GEMINI_MODEL")
    gemini_temperature: float = Field(default=0.7, validation_alias="GEMINI_TEMPERATURE")
    gemini_max_tokens: Optional[int] = Field(default=None, validation_alias="GEMINI_MAX_TOKENS")

    # Groq Config
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", validation_alias="GROQ_MODEL")
    groq_temperature: float = Field(default=0.7, validation_alias="GROQ_TEMPERATURE")
    groq_max_tokens: Optional[int] = Field(default=None, validation_alias="GROQ_MAX_TOKENS")

    @property
    def gemini(self) -> GeminiSettings:
        return GeminiSettings(
            api_key=self.gemini_api_key,
            model=self.gemini_model,
            temperature=self.gemini_temperature,
            max_output_tokens=self.gemini_max_tokens,
        )

    @property
    def groq(self) -> GroqSettings:
        return GroqSettings(
            api_key=self.groq_api_key,
            model=self.groq_model,
            temperature=self.groq_temperature,
            max_tokens=self.groq_max_tokens,
        )


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton instance of Settings."""
    return Settings()
