import os
import sys
import tempfile
from unittest.mock import MagicMock, patch
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.settings import settings
from src.core.exceptions import ImageToTextError
from src.modules.image.image_to_text import ImageToText, get_image_to_text


@pytest.fixture
def mock_groq_client():
    """Mock Groq client with chat.completions.create capability."""
    client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "This is a detailed description of the image."
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    client.chat.completions.create.return_value = mock_response
    return client


def test_init_raises_value_error_when_api_key_missing():
    """Test that missing GROQ_API_KEY raises ValueError."""
    with patch.dict(os.environ, {"GROQ_API_KEY": ""}, clear=False), \
         patch("src.modules.image.image_to_text.settings") as mock_settings:
        mock_settings.GROQ_API_KEY = ""
        mock_settings.groq_api_key = ""

        with pytest.raises(ValueError) as exc_info:
            ImageToText()

        assert "Missing required environment variables: GROQ_API_KEY" in str(exc_info.value)


def test_init_success_with_key():
    """Test that ImageToText initializes when GROQ_API_KEY is available."""
    with patch.dict(os.environ, {"GROQ_API_KEY": "gsk_dummy_key"}, clear=False):
        itt = ImageToText()
        assert itt is not None


def test_client_singleton_property(mock_groq_client):
    """Test that client property creates and caches Groq client."""
    itt = ImageToText(client=mock_groq_client)
    assert itt.client is mock_groq_client


@pytest.mark.asyncio
async def test_analyze_image_with_bytes(mock_groq_client):
    """Test analyzing image with binary bytes."""
    itt = ImageToText(client=mock_groq_client)
    image_bytes = b"fake_image_binary_data"

    result = await itt.analyze_image(image_bytes)

    assert result == "This is a detailed description of the image."
    mock_groq_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_groq_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == settings.ITT_MODEL_NAME
    assert call_kwargs["max_tokens"] == 1000

    messages = call_kwargs["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0]["text"] == "Please describe what you see in this image in detail."
    assert messages[0]["content"][1]["type"] == "image_url"
    assert "data:image/jpeg;base64," in messages[0]["content"][1]["image_url"]["url"]


@pytest.mark.asyncio
async def test_analyze_image_with_file_path(mock_groq_client):
    """Test analyzing image from an existing file path."""
    itt = ImageToText(client=mock_groq_client)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        tmp_file.write(b"sample_jpeg_content")
        tmp_path = tmp_file.name

    try:
        result = await itt.analyze_image(tmp_path, prompt="What objects are in this picture?")
        assert result == "This is a detailed description of the image."

        call_kwargs = mock_groq_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["content"][0]["text"] == "What objects are in this picture?"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@pytest.mark.asyncio
async def test_analyze_image_nonexistent_file_raises_error(mock_groq_client):
    """Test that a non-existent file path raises ImageToTextError wrapping ValueError."""
    itt = ImageToText(client=mock_groq_client)

    with pytest.raises(ImageToTextError) as exc_info:
        await itt.analyze_image("non_existent_file_12345.jpg")

    assert "Image file not found" in str(exc_info.value)


@pytest.mark.asyncio
async def test_analyze_image_empty_bytes_raises_error(mock_groq_client):
    """Test that empty bytes raise ImageToTextError wrapping ValueError."""
    itt = ImageToText(client=mock_groq_client)

    with pytest.raises(ImageToTextError) as exc_info:
        await itt.analyze_image(b"")

    assert "Image data cannot be empty" in str(exc_info.value)


@pytest.mark.asyncio
async def test_analyze_image_no_response_choices_raises_error(mock_groq_client):
    """Test that empty response choices raise ImageToTextError."""
    mock_response = MagicMock()
    mock_response.choices = []
    mock_groq_client.chat.completions.create.return_value = mock_response

    itt = ImageToText(client=mock_groq_client)

    with pytest.raises(ImageToTextError) as exc_info:
        await itt.analyze_image(b"dummy_data")

    assert "No response received from the vision model" in str(exc_info.value)


@pytest.mark.asyncio
async def test_analyze_image_api_exception_raises_error(mock_groq_client):
    """Test that Groq API exceptions are wrapped in ImageToTextError."""
    mock_groq_client.chat.completions.create.side_effect = RuntimeError("Groq API 500 error")

    itt = ImageToText(client=mock_groq_client)

    with pytest.raises(ImageToTextError) as exc_info:
        await itt.analyze_image(b"dummy_data")

    assert "Failed to analyze image: Groq API 500 error" in str(exc_info.value)


def test_get_image_to_text_singleton():
    """Test singleton retrieval via get_image_to_text()."""
    with patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test"}, clear=False):
        inst1 = get_image_to_text()
        inst2 = get_image_to_text()
        assert inst1 is inst2


def test_settings_itt_model_configuration():
    """Test that ITT_MODEL_NAME is properly exposed on settings."""
    assert hasattr(settings, "itt_model_name")
    assert hasattr(settings, "ITT_MODEL_NAME")
    assert isinstance(settings.ITT_MODEL_NAME, str)
    assert len(settings.ITT_MODEL_NAME) > 0
    assert settings.ITT_MODEL_NAME == settings.itt_model_name

