"""Speech-to-text module alias forwarding to src.modules.speech.speech_to_text."""
from src.modules.speech.speech_to_text import (
    SpeechToText,
    SpeechToTextError,
    get_speech_to_text,
)

__all__ = [
    "SpeechToText",
    "SpeechToTextError",
    "get_speech_to_text",
]
