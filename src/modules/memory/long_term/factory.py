from typing import Callable, Dict, Optional, Union

from src.config.settings import MemoryProviderType, Settings, get_settings
from src.modules.memory.long_term.interface import BaseVectorStore
from src.modules.memory.long_term.provider.mongodb import MongoDBVectorStore
from src.modules.memory.long_term.provider.qdrant import QdrantVectorStore


class MemoryFactory:
    """Factory for creating and retrieving vector store memory providers."""

    _registry: Dict[str, Callable[[Settings], BaseVectorStore]] = {
        MemoryProviderType.QDRANT.value: lambda s: QdrantVectorStore(s.qdrant),
        MemoryProviderType.MONGODB.value: lambda s: MongoDBVectorStore(s.mongodb),
    }

    _instances: Dict[str, BaseVectorStore] = {}

    @classmethod
    def register_provider(
        cls,
        provider_name: str,
        creator: Callable[[Settings], BaseVectorStore],
    ) -> None:
        """Register a new memory provider type and its factory function."""
        cls._registry[provider_name.lower()] = creator

    @classmethod
    def create(
        cls,
        provider: Optional[Union[MemoryProviderType, str]] = None,
        settings: Optional[Settings] = None,
        force_new: bool = False,
    ) -> BaseVectorStore:
        """Create or return a singleton memory provider instance.

        If no provider is specified, it uses the active provider configured in settings (.env).

        Args:
            provider: Optional explicit provider name or enum (e.g. 'qdrant', 'mongodb').
            settings: Optional custom Settings instance.
            force_new: If True, bypass cached instance and create a new one.

        Returns:
            An instance of BaseVectorStore.

        Raises:
            ValueError: If the requested provider is not supported or registered.
        """
        app_settings = settings or get_settings()

        if provider is None:
            provider_key = (
                app_settings.memory_provider.value
                if isinstance(app_settings.memory_provider, MemoryProviderType)
                else str(app_settings.memory_provider)
            )
        elif isinstance(provider, MemoryProviderType):
            provider_key = provider.value
        else:
            provider_key = str(provider).lower()

        provider_key = provider_key.lower()

        if not force_new and provider_key in cls._instances:
            return cls._instances[provider_key]

        creator = cls._registry.get(provider_key)
        if not creator:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unsupported Memory provider: '{provider_key}'. Available providers: {available}"
            )

        instance = creator(app_settings)
        if not force_new:
            cls._instances[provider_key] = instance
        return instance


# Aliases for convenience
VectorStoreFactory = MemoryFactory


def get_vector_store(
    provider: Optional[Union[MemoryProviderType, str]] = None,
    settings: Optional[Settings] = None,
) -> BaseVectorStore:
    """Convenience function to get the configured vector store memory provider."""
    return MemoryFactory.create(provider=provider, settings=settings)
