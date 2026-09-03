import os
import sys
from unittest.mock import MagicMock, patch
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.exceptions import TextToSpeechError
from src.modules.speech.text_to_speech import TextToSpeech, get_text_to_speech


@pytest.fixture
def mock_elevenlabs_client():
    """Mock ElevenLabs client with text_to_speech.convert capability."""
    client = MagicMock()
    # Mock text_to_speech.convert returning a generator of chunks
    client.text_to_speech.convert.return_value = [b"chunk1_", b"chunk2_", b"audio_bytes"]
    return client


def test_init_raises_value_error_when_env_vars_missing():
    """Test that missing required environment variables raise ValueError."""
    with patch.dict(os.environ, {}, clear=True), \
         patch("src.modules.speech.text_to_speech.settings") as mock_settings:
        mock_settings.ELEVENLABS_API_KEY = ""
        mock_settings.ELEVENLABS_VOICE_ID = ""
        mock_settings.elevenlabs_api_key = ""
        mock_settings.elevenlabs_voice_id = ""

        with pytest.raises(ValueError) as exc_info:
            TextToSpeech()

        assert "Missing required environment variables" in str(exc_info.value)
        assert "ELEVENLABS_API_KEY" in str(exc_info.value)
        assert "ELEVENLABS_VOICE_ID" in str(exc_info.value)


def test_init_success_with_constructor_args():
    """Test that passing credentials to constructor succeeds without environment variables."""
    tts = TextToSpeech(
        api_key="test_api_key",
        voice_id="test_voice_id",
    )
    assert tts._api_key == "test_api_key"
    assert tts._voice_id == "test_voice_id"


def test_client_singleton_property(mock_elevenlabs_client):
    """Test that client property instantiates ElevenLabs client lazily as singleton."""
    with patch("src.modules.speech.text_to_speech.ElevenLabs") as mock_cls:
        mock_cls.return_value = mock_elevenlabs_client

        tts = TextToSpeech(
            api_key="test_api_key",
            voice_id="test_voice_id",
        )

        client1 = tts.client
        client2 = tts.client

        assert client1 is mock_elevenlabs_client
        assert client2 is mock_elevenlabs_client
        mock_cls.assert_called_once_with(api_key="test_api_key")


@pytest.mark.asyncio
async def test_synthesize_success(mock_elevenlabs_client):
    """Test synthesizing valid text returns concatenated audio bytes."""
    tts = TextToSpeech(
        api_key="test_api_key",
        voice_id="voice_xyz",
        client=mock_elevenlabs_client,
    )

    audio = await tts.synthesize("Hello world, this is a test speech synthesis.")

    assert audio == b"chunk1_chunk2_audio_bytes"
    mock_elevenlabs_client.text_to_speech.convert.assert_called_once()
    call_kwargs = mock_elevenlabs_client.text_to_speech.convert.call_args.kwargs
    assert call_kwargs["voice_id"] == "voice_xyz"
    assert call_kwargs["text"] == "Hello world, this is a test speech synthesis."
    assert call_kwargs["voice_settings"].stability == 0.5
    assert call_kwargs["voice_settings"].similarity_boost == 0.5


@pytest.mark.asyncio
async def test_synthesize_empty_text_raises_value_error(mock_elevenlabs_client):
    """Test that empty or whitespace-only input raises ValueError."""
    tts = TextToSpeech(api_key="key", voice_id="voice", client=mock_elevenlabs_client)

    with pytest.raises(ValueError, match="Input text cannot be empty"):
        await tts.synthesize("")

    with pytest.raises(ValueError, match="Input text cannot be empty"):
        await tts.synthesize("   \n\t  ")


@pytest.mark.asyncio
async def test_synthesize_text_too_long_raises_value_error(mock_elevenlabs_client):
    """Test that text exceeding 5000 characters raises ValueError."""
    tts = TextToSpeech(api_key="key", voice_id="voice", client=mock_elevenlabs_client)

    long_text = "a" * 5001
    with pytest.raises(ValueError, match="exceeds maximum length of 5000 characters"):
        await tts.synthesize(long_text)


@pytest.mark.asyncio
async def test_synthesize_empty_audio_raises_tts_error(mock_elevenlabs_client):
    """Test that empty audio returned by the generator raises TextToSpeechError."""
    mock_elevenlabs_client.text_to_speech.convert.return_value = [b""]
    tts = TextToSpeech(api_key="key", voice_id="voice", client=mock_elevenlabs_client)

    with pytest.raises(TextToSpeechError, match="Generated audio is empty"):
        await tts.synthesize("Some valid text")


@pytest.mark.asyncio
async def test_synthesize_api_exception_raises_tts_error(mock_elevenlabs_client):
    """Test that API exceptions during synthesis are wrapped in TextToSpeechError."""
    mock_elevenlabs_client.text_to_speech.convert.side_effect = RuntimeError("ElevenLabs quota exceeded")
    tts = TextToSpeech(api_key="key", voice_id="voice", client=mock_elevenlabs_client)

    with pytest.raises(TextToSpeechError) as exc_info:
        await tts.synthesize("Some valid text")

    assert "Text-to-speech conversion failed: ElevenLabs quota exceeded" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_get_text_to_speech_helper(mock_elevenlabs_client):
    """Test get_text_to_speech helper factory function."""
    tts = get_text_to_speech(
        api_key="key",
        voice_id="voice",
        client=mock_elevenlabs_client,
    )
    assert isinstance(tts, TextToSpeech)
    assert tts.client is mock_elevenlabs_client
