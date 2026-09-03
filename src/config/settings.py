from enum import Enum
from functools import lru_cache
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProviderType(str, Enum):
    GEMINI = "gemini"
    GROQ = "groq"


class MemoryProviderType(str, Enum):
    QDRANT = "qdrant"
    MONGODB = "mongodb"


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


class QdrantSettings(BaseModel):
    """Configuration settings for Qdrant Vector Database."""
    url: str = Field(default="http://localhost:6333", description="Qdrant service URL")
    api_key: Optional[str] = Field(default=None, description="Qdrant API Key (for cloud/auth)")
    collection_name: str = Field(default="long_term_memory", description="Collection name for long-term memory")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Sentence Transformers embedding model name")
    similarity_threshold: float = Field(default=0.9, ge=0.0, le=1.0, description="Threshold for considering memories similar")


class MongoDBSettings(BaseModel):
    """Configuration settings for MongoDB Vector Database."""
    uri: str = Field(default="mongodb://localhost:27017", description="MongoDB connection URI")
    database: str = Field(default="oria_agent", description="MongoDB database name")
    collection_name: str = Field(default="long_term_memory", description="MongoDB collection name")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Sentence Transformers embedding model name")
    similarity_threshold: float = Field(default=0.9, ge=0.0, le=1.0, description="Threshold for considering memories similar")


class ElevenLabsSettings(BaseModel):
    """Configuration settings for ElevenLabs Text-to-Speech."""
    api_key: str = Field(default="", description="ElevenLabs API Key")
    voice_id: str = Field(default="", description="Default ElevenLabs Voice ID")


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

    # Active Memory / Vector DB provider
    memory_provider: MemoryProviderType = Field(
        default=MemoryProviderType.QDRANT,
        validation_alias="MEMORY_PROVIDER",
        description="The active memory provider (qdrant, mongodb)",
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

    # Qdrant Config
    qdrant_url: str = Field(default="http://localhost:6333", validation_alias="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, validation_alias="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(default="long_term_memory", validation_alias="QDRANT_COLLECTION_NAME")
    qdrant_embedding_model: str = Field(default="all-MiniLM-L6-v2", validation_alias="EMBEDDING_MODEL")
    qdrant_similarity_threshold: float = Field(default=0.9, validation_alias="SIMILARITY_THRESHOLD")

    # MongoDB Config
    mongodb_uri: str = Field(default="mongodb://localhost:27017", validation_alias="MONGODB_URI")
    mongodb_database: str = Field(default="oria_agent", validation_alias="MONGODB_DATABASE")
    mongodb_collection_name: str = Field(default="long_term_memory", validation_alias="MONGODB_COLLECTION_NAME")
    mongodb_embedding_model: str = Field(default="all-MiniLM-L6-v2", validation_alias="MONGODB_EMBEDDING_MODEL")
    mongodb_similarity_threshold: float = Field(default=0.9, validation_alias="MONGODB_SIMILARITY_THRESHOLD")

    # Memory Configuration
    memory_top_k: int = Field(
        default=5,
        validation_alias="MEMORY_TOP_K",
        description="Number of top relevant memories to retrieve",
    )
    small_text_model_name: str = Field(
        default="openai/gpt-oss-20b",
        validation_alias="SMALL_TEXT_MODEL_NAME",
        description="Fast/small model for memory analysis",
    )

    # ElevenLabs / Text-To-Speech Configuration
    elevenlabs_api_key: str = Field(default="", validation_alias="ELEVENLABS_API_KEY")
    elevenlabs_voice_id: str = Field(default="", validation_alias="ELEVENLABS_VOICE_ID")

    @property
    def GROQ_API_KEY(self) -> str:
        return self.groq_api_key

    @property
    def SMALL_TEXT_MODEL_NAME(self) -> str:
        return self.small_text_model_name

    @property
    def MEMORY_TOP_K(self) -> int:
        return self.memory_top_k

    @property
    def ELEVENLABS_API_KEY(self) -> str:
        return self.elevenlabs_api_key

    @property
    def ELEVENLABS_VOICE_ID(self) -> str:
        return self.elevenlabs_voice_id

    @property
    def elevenlabs(self) -> ElevenLabsSettings:
        return ElevenLabsSettings(
            api_key=self.elevenlabs_api_key,
            voice_id=self.elevenlabs_voice_id,
        )

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

    @property
    def qdrant(self) -> QdrantSettings:
        return QdrantSettings(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            collection_name=self.qdrant_collection_name,
            embedding_model=self.qdrant_embedding_model,
            similarity_threshold=self.qdrant_similarity_threshold,
        )

    @property
    def mongodb(self) -> MongoDBSettings:
        return MongoDBSettings(
            uri=self.mongodb_uri,
            database=self.mongodb_database,
            collection_name=self.mongodb_collection_name,
            embedding_model=self.mongodb_embedding_model,
            similarity_threshold=self.mongodb_similarity_threshold,
        )


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton instance of Settings."""
    return Settings()


# Singleton instance for convenience
settings = get_settings()
