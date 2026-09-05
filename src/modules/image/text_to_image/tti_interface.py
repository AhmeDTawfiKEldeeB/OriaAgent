from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import BaseModel, Field


class ScenarioPrompt(BaseModel):
    """Class for the scenario response."""

    narrative: str = Field(..., description="The AI's narrative response to the question")
    image_prompt: str = Field(..., description="The visual prompt to generate an image representing the scene")


class EnhancedPrompt(BaseModel):
    """Class for the text prompt."""

    content: str = Field(
        ...,
        description="The enhanced text prompt to generate an image",
    )


class TTIProvider(ABC):
    """Abstract Base Class for Text-to-Image (TTI) providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the TTI provider."""
        pass

    @abstractmethod
    async def generate_image(self, prompt: str, output_path: str = "") -> bytes:
        """Generate an image from a prompt and optionally save it to output_path.

        Args:
            prompt: Visual prompt describing the image to generate.
            output_path: Optional file path to save the generated image.

        Returns:
            bytes: The raw generated image data.

        Raises:
            ValueError: If prompt is empty.
            TextToImageError: If image generation fails.
        """
        pass

    @abstractmethod
    async def create_scenario(self, chat_history: Optional[List] = None) -> ScenarioPrompt:
        """Create a first-person narrative scenario and corresponding image prompt based on chat history.

        Args:
            chat_history: Optional list of previous chat messages.

        Returns:
            ScenarioPrompt: Generated narrative and image prompt.
        """
        pass

    @abstractmethod
    async def enhance_prompt(self, prompt: str) -> str:
        """Enhance a simple prompt with additional details and context.

        Args:
            prompt: Base prompt to be enhanced.

        Returns:
            str: Enhanced prompt.
        """
        pass
