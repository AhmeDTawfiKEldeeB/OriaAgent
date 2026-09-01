from typing import Callable, Optional, Union

from langchain_core.language_models.chat_models import BaseChatModel

from src.config.settings import LLMProviderType, Settings, get_settings
from src.infrastructure.llm.interface import BaseLLMProvider
from src.infrastructure.llm.provider.gemini_provider import GeminiProvider
from src.infrastructure.llm.provider.groq_provider import GroqProvider


class LLMFactory:
    """Factory for creating LLM provider instances based on configuration."""

    _registry: dict[str, Callable[[Settings], BaseLLMProvider]] = {
        LLMProviderType.GEMINI.value: lambda s: GeminiProvider(s.gemini),
        LLMProviderType.GROQ.value: lambda s: GroqProvider(s.groq),
    }

    @classmethod
    def register_provider(
        cls,
        provider_name: str,
        creator: Callable[[Settings], BaseLLMProvider],
    ) -> None:
        """Register a new LLM provider type and its factory function."""
        cls._registry[provider_name.lower()] = creator

    @classmethod
    def create(
        cls,
        provider: Optional[Union[LLMProviderType, str]] = None,
        settings: Optional[Settings] = None,
    ) -> BaseLLMProvider:
        """Create and return an LLM provider instance.

        If no provider is specified, it uses the active provider configured in settings (from .env).

        Args:
            provider: Optional explicit provider name or enum (e.g. 'gemini', 'groq').
            settings: Optional custom Settings instance. Defaults to get_settings().

        Returns:
            An instance of BaseLLMProvider.

        Raises:
            ValueError: If the requested provider is not supported or registered.
        """
        app_settings = settings or get_settings()

        if provider is None:
            provider_key = (
                app_settings.llm_provider.value
                if isinstance(app_settings.llm_provider, LLMProviderType)
                else str(app_settings.llm_provider)
            )
        elif isinstance(provider, LLMProviderType):
            provider_key = provider.value
        else:
            provider_key = str(provider).lower()

        provider_key = provider_key.lower()

        creator = cls._registry.get(provider_key)
        if not creator:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unsupported LLM provider: '{provider_key}'. Available providers: {available}"
            )

        return creator(app_settings)


def get_llm_provider(
    provider: Optional[Union[LLMProviderType, str]] = None,
    settings: Optional[Settings] = None,
) -> BaseLLMProvider:
    """Convenience function to get an LLM provider instance from the factory."""
    return LLMFactory.create(provider=provider, settings=settings)


def get_chat_model(
    provider: Optional[Union[LLMProviderType, str]] = None,
    settings: Optional[Settings] = None,
) -> BaseChatModel:
    """Convenience function to get the underlying LangChain BaseChatModel directly."""
    return get_llm_provider(provider=provider, settings=settings).chat_model
