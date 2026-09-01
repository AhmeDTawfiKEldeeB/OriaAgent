from langchain_google_genai import ChatGoogleGenerativeAI

from src.config.settings import GeminiSettings
from src.infrastructure.llm.interface import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM Provider using LangChain's ChatGoogleGenerativeAI."""

    def __init__(self, settings: GeminiSettings):
        self._settings = settings
        if not self._settings.api_key:
            raise ValueError(
                "Gemini API key is required. Please set GEMINI_API_KEY in your .env file."
            )
        self._chat_model = ChatGoogleGenerativeAI(
            model=self._settings.model,
            api_key=self._settings.api_key,
            temperature=self._settings.temperature,
            max_output_tokens=self._settings.max_output_tokens,
        )

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._settings.model

    @property
    def chat_model(self) -> ChatGoogleGenerativeAI:
        return self._chat_model
