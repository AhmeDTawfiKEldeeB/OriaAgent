from typing import Callable, Dict, Optional, Union

from src.config.settings import Settings, TTIProviderType, get_settings
from src.modules.image.text_to_image.providers.cloudflare import CloudflareTTIProvider
from src.modules.image.text_to_image.tti_interface import TTIProvider


class TTIFactory:
    """Factory for creating and managing Text-to-Image provider instances."""

    _registry: Dict[str, Callable[[Settings], TTIProvider]] = {
        TTIProviderType.CLOUDFLARE.value: lambda s: CloudflareTTIProvider(
            account_id=s.CLOUDFLARE_ACCOUNT_ID,
            api_token=s.CLOUDFLARE_API_TOKEN,
            model=s.TTI_MODEL_NAME,
        ),
    }

    _instances: Dict[str, TTIProvider] = {}

    @classmethod
    def register_provider(
        cls,
        provider_name: str,
        creator: Callable[[Settings], TTIProvider],
    ) -> None:
        """Register a new TTI provider type and its factory function."""
        cls._registry[provider_name.lower()] = creator

    @classmethod
    def create(
        cls,
        provider: Optional[Union[TTIProviderType, str]] = None,
        settings: Optional[Settings] = None,
        force_new: bool = False,
    ) -> TTIProvider:
        """Create and return a TTI provider instance.

        If no provider is specified, it uses the active provider configured in settings (settings.TTI_PROVIDER),
        defaulting to Cloudflare.

        Args:
            provider: Optional explicit provider name or enum (e.g. 'cloudflare').
            settings: Optional custom Settings instance. Defaults to get_settings().
            force_new: If True, bypass cached instance and create a new one.

        Returns:
            An instance of TTIProvider.

        Raises:
            ValueError: If the requested provider is not supported or registered.
        """
        app_settings = settings or get_settings()

        if provider is None:
            if hasattr(app_settings, "tti_provider"):
                provider_val = app_settings.tti_provider
                provider_key = provider_val.value if isinstance(provider_val, TTIProviderType) else str(provider_val)
            else:
                provider_key = getattr(app_settings, "TTI_PROVIDER", "cloudflare")
        elif isinstance(provider, TTIProviderType):
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
                f"Unsupported TTI provider: '{provider_key}'. Available providers: {available}"
            )

        instance = creator(app_settings)
        if not force_new:
            cls._instances[provider_key] = instance
        return instance


def get_tti_provider(
    provider: Optional[Union[TTIProviderType, str]] = None,
    settings: Optional[Settings] = None,
) -> TTIProvider:
    """Convenience function to get a TTI provider instance from the factory."""
    return TTIFactory.create(provider=provider, settings=settings)
