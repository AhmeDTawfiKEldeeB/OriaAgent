import re
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from src.infrastructure.llm import get_chat_model as base_get_chat_model
from src.modules.image import ImageToText, TextToImage, get_image_to_text, get_text_to_image
from src.modules.speech import TextToSpeech, get_text_to_speech


def get_chat_model(temperature: Optional[float] = None) -> BaseChatModel:
    """Retrieve the active chat model from OriaAgent's LLM Factory.

    If temperature is specified, it binds the temperature parameter to the model.
    """
    model = base_get_chat_model()
    if temperature is not None:
        return model.bind(temperature=temperature)
    return model


def get_text_to_speech_module() -> TextToSpeech:
    """Get the TextToSpeech module instance."""
    return get_text_to_speech()


def get_text_to_image_module() -> TextToImage:
    """Get the TextToImage module instance."""
    return get_text_to_image()


def get_image_to_text_module() -> ImageToText:
    """Get the ImageToText module instance."""
    return get_image_to_text()


def remove_asterisk_content(text: str) -> str:
    """Remove content between asterisks from the text."""
    return re.sub(r"\*.*?\*", "", text).strip()


class AsteriskRemovalParser(StrOutputParser):
    """Output parser that cleans out any asterisk-wrapped stage directions."""

    def parse(self, text: str) -> str:
        return remove_asterisk_content(super().parse(text))
