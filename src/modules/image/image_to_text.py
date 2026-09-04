import base64
import logging
import os
from typing import Optional, Union

from dotenv import load_dotenv

load_dotenv()

from src.config.settings import settings
from src.core.exceptions import ImageToTextError

from groq import Groq


class ImageToText:
    """A class to handle image-to-text conversion using Groq's vision capabilities."""

    REQUIRED_ENV_VARS = ["GROQ_API_KEY"]

    def __init__(self, client: Optional[Groq] = None):
        """Initialize the ImageToText class and validate environment variables."""
        self._client: Optional[Groq] = client
        if self._client is None:
            self._validate_env_vars()
        self.logger = logging.getLogger(__name__)

    def _validate_env_vars(self) -> None:
        """Validate that all required environment variables are set."""
        missing_vars = [
            var
            for var in self.REQUIRED_ENV_VARS
            if not (os.getenv(var) or getattr(settings, var, None) or getattr(settings, var.lower(), None))
        ]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    @property
    def client(self) -> Groq:
        """Get or create Groq client instance using singleton pattern."""
        if self._client is None:
            api_key = getattr(settings, "GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY")
            self._client = Groq(api_key=api_key)
        return self._client

    @client.setter
    def client(self, client: Groq) -> None:
        """Set the Groq client instance."""
        self._client = client

    async def analyze_image(self, image_data: Union[str, bytes], prompt: str = "") -> str:
        """Analyze an image using Groq's vision capabilities.

        Args:
            image_data: Either a file path (str) or binary image data (bytes)
            prompt: Optional prompt to guide the image analysis

        Returns:
            str: Description or analysis of the image

        Raises:
            ValueError: If the image data is empty or invalid
            ImageToTextError: If the image analysis fails
        """
        try:
            # Handle file path
            if isinstance(image_data, str):
                if not os.path.exists(image_data):
                    raise ValueError(f"Image file not found: {image_data}")
                with open(image_data, "rb") as f:
                    image_bytes = f.read()
            else:
                image_bytes = image_data

            if not image_bytes:
                raise ValueError("Image data cannot be empty")

            # Convert image to base64
            base64_image = base64.b64encode(image_bytes).decode("utf-8")

            # Default prompt if none provided
            if not prompt:
                prompt = "Please describe what you see in this image in detail."

            # Create the messages for the vision API
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ]

            # Make the API call
            response = self.client.chat.completions.create(
                model=settings.ITT_MODEL_NAME,
                messages=messages,
                max_tokens=1000,
            )

            if not response.choices:
                raise ImageToTextError("No response received from the vision model")

            description = response.choices[0].message.content
            self.logger.info(f"Generated image description: {description}")

            return description

        except Exception as e:
            raise ImageToTextError(f"Failed to analyze image: {str(e)}") from e


_image_to_text_instance: Optional[ImageToText] = None


def get_image_to_text() -> ImageToText:
    """Get or create singleton ImageToText instance."""
    global _image_to_text_instance
    if _image_to_text_instance is None:
        _image_to_text_instance = ImageToText()
    return _image_to_text_instance
