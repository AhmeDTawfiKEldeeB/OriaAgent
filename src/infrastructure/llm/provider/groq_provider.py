from langchain_groq import ChatGroq

from src.config.settings import GroqSettings
from src.infrastructure.llm.interface import BaseLLMProvider


class GroqProvider(BaseLLMProvider):
    """Groq LLM Provider using LangChain's ChatGroq."""

    def __init__(self, settings: GroqSettings):
        self._settings = settings
        if not self._settings.api_key:
            raise ValueError(
                "Groq API key is required. Please set GROQ_API_KEY in your .env file."
            )
        self._chat_model = ChatGroq(
            model=self._settings.model,
            api_key=self._settings.api_key,
            temperature=self._settings.temperature,
            max_tokens=self._settings.max_tokens,
        )

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def model_name(self) -> str:
        return self._settings.model

    @property
    def chat_model(self) -> ChatGroq:
        return self._chat_model
