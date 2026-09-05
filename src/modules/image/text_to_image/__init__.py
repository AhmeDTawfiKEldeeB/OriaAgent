import logging
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from src.modules.image.text_to_image.providers.cloudflare import CloudflareTTIProvider
from src.modules.image.text_to_image.tti_factory import TTIFactory, get_tti_provider
from src.modules.image.text_to_image.tti_interface import (
    EnhancedPrompt,
    ScenarioPrompt,
    TTIProvider,
)


class TextToImage:
    """A class to coordinate text-to-image generation and prompt assistance."""

    REQUIRED_ENV_VARS = ["GROQ_API_KEY"]

    def __init__(self, provider: Optional[TTIProvider] = None):
        """Initialize TextToImage with a TTI provider or default from factory."""
        self._provider = provider
        self.logger = logging.getLogger(__name__)

    @property
    def provider(self) -> TTIProvider:
        """Get or initialize the TTI provider from TTIFactory."""
        if self._provider is None:
            self._provider = TTIFactory.create()
        return self._provider

    @provider.setter
    def provider(self, provider: TTIProvider) -> None:
        """Set the active TTI provider."""
        self._provider = provider

    async def generate_image(self, prompt: str, output_path: str = "") -> bytes:
        """Generate an image from a prompt using the active provider.

        Args:
            prompt: Visual prompt describing the image to generate.
            output_path: Optional file path where the generated image will be saved.

        Returns:
            bytes: Binary content of the generated image.
        """
        return await self.provider.generate_image(prompt=prompt, output_path=output_path)

    async def create_scenario(self, chat_history: list = None) -> ScenarioPrompt:
        """Create a first-person narrative scenario and corresponding image prompt based on chat history."""
        return await self.provider.create_scenario(chat_history=chat_history)

    async def enhance_prompt(self, prompt: str) -> str:
        """Enhance a simple prompt with additional details and context."""
        return await self.provider.enhance_prompt(prompt=prompt)


_text_to_image_instance: Optional[TextToImage] = None


def get_text_to_image(provider: Optional[TTIProvider] = None) -> TextToImage:
    """Get or create singleton TextToImage instance."""
    global _text_to_image_instance
    if _text_to_image_instance is None:
        _text_to_image_instance = TextToImage(provider=provider)
    elif provider is not None:
        _text_to_image_instance.provider = provider
    return _text_to_image_instance


__all__ = [
    "TTIProvider",
    "TTIFactory",
    "get_tti_provider",
    "CloudflareTTIProvider",
    "TextToImage",
    "get_text_to_image",
    "ScenarioPrompt",
    "EnhancedPrompt",
]
