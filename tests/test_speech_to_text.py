import os
import sys
from unittest.mock import MagicMock, patch
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.exceptions import SpeechToTextError
from src.modules.speech.speech_to_text import SpeechToText, get_speech_to_text


@pytest.fixture
def mock_groq_client():
    """Mock Groq client with audio.transcriptions.create capability."""
    client = MagicMock()
    mock_transcription = MagicMock()
    mock_transcription.text = "Hello, this is a test transcription from Groq Whisper."
    client.audio.transcriptions.create.return_value = mock_transcription
    return client


def test_init_raises_value_error_when_api_key_missing():
    """Test that missing GROQ_API_KEY raises ValueError."""
    with patch.dict(os.environ, {}, clear=True), \
         patch("src.modules.speech.speech_to_text.settings") as mock_settings:
        mock_settings.GROQ_API_KEY = ""
        mock_settings.groq_api_key = ""

        with pytest.raises(ValueError) as exc_info:
            SpeechToText()

        assert "Missing required environment variables" in str(exc_info.value)
        assert "GROQ_API_KEY" in str(exc_info.value)


def test_init_success_with_constructor_args():
    """Test that passing credentials to constructor succeeds without environment variables."""
    stt = SpeechToText(
        api_key="gsk_test_api_key",
        model_name="whisper-large-v3-turbo",
        language="en",
    )
    assert stt._api_key == "gsk_test_api_key"
    assert stt._model_name == "whisper-large-v3-turbo"
    assert stt._language == "en"


def test_client_singleton_property(mock_groq_client):
    """Test that client property instantiates Groq client lazily as singleton."""
    with patch("src.modules.speech.speech_to_text.Groq") as mock_cls:
        mock_cls.return_value = mock_groq_client

        stt = SpeechToText(api_key="gsk_test_api_key")

        client1 = stt.client
        client2 = stt.client

        assert client1 is mock_groq_client
        assert client2 is mock_groq_client
        mock_cls.assert_called_once_with(api_key="gsk_test_api_key")


@pytest.mark.asyncio
async def test_transcribe_success(mock_groq_client):
    """Test transcribing valid audio bytes returns text."""
    stt = SpeechToText(
        api_key="gsk_test_key",
        model_name="whisper-large-v3-turbo",
        language="en",
        client=mock_groq_client,
    )

    audio_data = b"RIFF....WAVEfmt ....data...."
    result = await stt.transcribe(audio_data)

    assert result == "Hello, this is a test transcription from Groq Whisper."
    mock_groq_client.audio.transcriptions.create.assert_called_once()
    call_kwargs = mock_groq_client.audio.transcriptions.create.call_args.kwargs
    assert call_kwargs["model"] == "whisper-large-v3-turbo"
    assert call_kwargs["language"] == "en"
    assert call_kwargs["response_format"] == "text"


@pytest.mark.asyncio
async def test_transcribe_string_response(mock_groq_client):
    """Test transcribing when API returns string directly instead of object with .text."""
    mock_groq_client.audio.transcriptions.create.return_value = "Direct string response."
    stt = SpeechToText(api_key="gsk_test_key", client=mock_groq_client)

    result = await stt.transcribe(b"dummy audio data")
    assert result == "Direct string response."


@pytest.mark.asyncio
async def test_transcribe_empty_audio_raises_value_error(mock_groq_client):
    """Test that empty audio bytes raises ValueError."""
    stt = SpeechToText(api_key="gsk_test_key", client=mock_groq_client)

    with pytest.raises(ValueError, match="Audio data cannot be empty"):
        await stt.transcribe(b"")


@pytest.mark.asyncio
async def test_transcribe_empty_result_raises_stt_error(mock_groq_client):
    """Test that empty transcription result raises SpeechToTextError."""
    mock_groq_client.audio.transcriptions.create.return_value = "   "
    stt = SpeechToText(api_key="gsk_test_key", client=mock_groq_client)

    with pytest.raises(SpeechToTextError, match="Transcription result is empty"):
        await stt.transcribe(b"dummy audio data")


@pytest.mark.asyncio
async def test_transcribe_api_error_raises_stt_error(mock_groq_client):
    """Test that API exceptions during transcription are wrapped in SpeechToTextError."""
    mock_groq_client.audio.transcriptions.create.side_effect = RuntimeError("Groq rate limit exceeded")
    stt = SpeechToText(api_key="gsk_test_key", client=mock_groq_client)

    with pytest.raises(SpeechToTextError) as exc_info:
        await stt.transcribe(b"dummy audio data")

    assert "Speech-to-text conversion failed: Groq rate limit exceeded" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)


@pytest.mark.asyncio
async def test_transcribe_cleans_up_temp_file(mock_groq_client):
    """Test that temporary .wav file is cleaned up after transcription."""
    created_file = None

    def fake_create(**kwargs):
        nonlocal created_file
        created_file = kwargs["file"].name
        assert os.path.exists(created_file)
        return "Cleaned up test."

    mock_groq_client.audio.transcriptions.create.side_effect = fake_create
    stt = SpeechToText(api_key="gsk_test_key", client=mock_groq_client)

    result = await stt.transcribe(b"some audio data")
    assert result == "Cleaned up test."
    assert created_file is not None
    assert not os.path.exists(created_file)


def test_get_speech_to_text_helper(mock_groq_client):
    """Test get_speech_to_text factory helper function."""
    stt = get_speech_to_text(api_key="gsk_test_key", client=mock_groq_client)
    assert isinstance(stt, SpeechToText)
    assert stt.client is mock_groq_client
