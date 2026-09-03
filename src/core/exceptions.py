"""Core application exceptions."""


class OriaAgentException(Exception):
    """Base exception for all OriaAgent exceptions."""
    pass


class TextToSpeechError(OriaAgentException):
    """Raised when text-to-speech synthesis fails."""
    pass


class SpeechToTextError(OriaAgentException):
    """Raised when speech-to-text transcription fails."""
    pass
