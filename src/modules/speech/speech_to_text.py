import os
import tempfile
from typing import Optional

from groq import Groq

from src.config.settings import settings
from src.core.exceptions import SpeechToTextError


class SpeechToText:
    """A class to handle speech-to-text conversion using Groq's Whisper model."""

    # Required environment variables
    REQUIRED_ENV_VARS = ["GROQ_API_KEY"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "whisper-large-v3-turbo",
        client: Optional[Groq] = None,
        language: str = "en",
    ):
        """Initialize the SpeechToText class and validate environment variables."""
        self._api_key = api_key
        self._model_name = model_name
        self._language = language
        self._client: Optional[Groq] = client

        # Validate environment variables if client or credentials not explicitly provided
        self._validate_env_vars()

    def _validate_env_vars(self) -> None:
        """Validate that all required environment variables are set."""
        missing_vars = []
        for var in self.REQUIRED_ENV_VARS:
            val = (
                (self._api_key if var == "GROQ_API_KEY" else None)
                or os.getenv(var)
                or getattr(settings, var, None)
                or getattr(settings, var.lower(), None)
            )
            if not val:
                missing_vars.append(var)

        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    @property
    def client(self) -> Groq:
        """Get or create Groq client instance using singleton pattern."""
        if self._client is None:
            api_key = (
                self._api_key
                or os.getenv("GROQ_API_KEY")
                or getattr(settings, "GROQ_API_KEY", "")
            )
            self._client = Groq(api_key=api_key)
        return self._client

    async def transcribe(self, audio_data: bytes) -> str:
        """Convert speech to text using Groq's Whisper model.

        Args:
            audio_data: Binary audio data

        Returns:
            str: Transcribed text

        Raises:
            ValueError: If the audio file is empty or invalid
            SpeechToTextError: If the transcription fails
        """
        if not audio_data:
            raise ValueError("Audio data cannot be empty")

        temp_file_path = None
        try:
            # Create a temporary file with .wav extension
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name

            try:
                # Open the temporary file for the API request
                with open(temp_file_path, "rb") as audio_file:
                    transcription = self.client.audio.transcriptions.create(
                        file=audio_file,
                        model=self._model_name,
                        language=self._language,
                        response_format="text",
                    )

                if hasattr(transcription, "text"):
                    result_text = transcription.text
                else:
                    result_text = str(transcription)

                if not result_text or not result_text.strip():
                    raise SpeechToTextError("Transcription result is empty")

                return result_text.strip()

            finally:
                # Clean up the temporary file safely
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                    except Exception:
                        pass

        except Exception as e:
            if isinstance(e, (ValueError, SpeechToTextError)):
                raise
            raise SpeechToTextError(f"Speech-to-text conversion failed: {str(e)}") from e


def get_speech_to_text(
    api_key: Optional[str] = None,
    model_name: str = "whisper-large-v3-turbo",
    client: Optional[Groq] = None,
    language: str = "en",
) -> SpeechToText:
    """Get a SpeechToText instance."""
    return SpeechToText(
        api_key=api_key,
        model_name=model_name,
        client=client,
        language=language,
    )
