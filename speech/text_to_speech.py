"""Text-to-speech module alias forwarding to src.modules.speech.text_to_speech."""
from src.modules.speech.text_to_speech import (
    TextToSpeech,
    TextToSpeechError,
    get_text_to_speech,
)

__all__ = [
    "TextToSpeech",
    "TextToSpeechError",
    "get_text_to_speech",
]
