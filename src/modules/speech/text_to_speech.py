import os
from typing import Optional

from elevenlabs import ElevenLabs, Voice, VoiceSettings

from src.config.settings import settings
from src.core.exceptions import TextToSpeechError


class TextToSpeech:
    """A class to handle text-to-speech conversion using ElevenLabs."""

    # Required environment variables
    REQUIRED_ENV_VARS = ["ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: Optional[str] = None,
        client: Optional[ElevenLabs] = None,
    ):
        """Initialize the TextToSpeech class and validate environment variables."""
        self._api_key = api_key
        self._voice_id = voice_id
        self._client: Optional[ElevenLabs] = client

        # Validate environment variables if client or credentials not explicitly provided
        self._validate_env_vars()

    def _validate_env_vars(self) -> None:
        """Validate that all required environment variables are set."""
        missing_vars = []
        for var in self.REQUIRED_ENV_VARS:
            val = (
                (self._api_key if var == "ELEVENLABS_API_KEY" else None)
                or (self._voice_id if var == "ELEVENLABS_VOICE_ID" else None)
                or os.getenv(var)
                or getattr(settings, var, None)
                or getattr(settings, var.lower(), None)
            )
            if not val:
                missing_vars.append(var)

        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    @property
    def client(self) -> ElevenLabs:
        """Get or create ElevenLabs client instance using singleton pattern."""
        if self._client is None:
            api_key = (
                self._api_key
                or os.getenv("ELEVENLABS_API_KEY")
                or getattr(settings, "ELEVENLABS_API_KEY", "")
            )
            self._client = ElevenLabs(api_key=api_key)
        return self._client

    async def synthesize(self, text: str) -> bytes:
        """Convert text to speech using ElevenLabs.

        Args:
            text: Text to convert to speech

        Returns:
            bytes: Audio data

        Raises:
            ValueError: If the input text is empty or too long
            TextToSpeechError: If the text-to-speech conversion fails
        """
        if not text.strip():
            raise ValueError("Input text cannot be empty")

        if len(text) > 5000:  # ElevenLabs typical limit
            raise ValueError("Input text exceeds maximum length of 5000 characters")

        try:
            voice_id = (
                self._voice_id
                or os.getenv("ELEVENLABS_VOICE_ID")
                or getattr(settings, "ELEVENLABS_VOICE_ID", "")
            )

            audio_generator = self.client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                voice_settings=VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.5,
                ),
            )

            # Convert generator to bytes
            if isinstance(audio_generator, (bytes, bytearray)):
                audio_bytes = bytes(audio_generator)
            else:
                audio_bytes = b"".join(audio_generator)

            if not audio_bytes:
                raise TextToSpeechError("Generated audio is empty")

            return audio_bytes

        except Exception as e:
            if isinstance(e, TextToSpeechError):
                raise
            raise TextToSpeechError(f"Text-to-speech conversion failed: {str(e)}") from e


def get_text_to_speech(
    api_key: Optional[str] = None,
    voice_id: Optional[str] = None,
    client: Optional[ElevenLabs] = None,
) -> TextToSpeech:
    """Get a TextToSpeech instance."""
    return TextToSpeech(
        api_key=api_key,
        voice_id=voice_id,
        client=client,
    )
