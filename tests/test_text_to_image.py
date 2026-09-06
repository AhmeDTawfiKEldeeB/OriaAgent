import base64
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.settings import settings, TTIProviderType
from src.core.exceptions import TextToImageError
from src.core.prompts import IMAGE_ENHANCEMENT_PROMPT, IMAGE_SCENARIO_PROMPT
from src.modules.image.text_to_image import (
    CloudflareTTIProvider,
    EnhancedPrompt,
    ScenarioPrompt,
    TTIFactory,
    TTIProvider,
    TextToImage,
    get_text_to_image,
    get_tti_provider,
)


@pytest.fixture
def mock_cloudflare_client():
    """Mock HTTP client for Cloudflare Workers AI."""
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "image/png"}
    mock_response.content = b"fake_image_bytes"
    client.post.return_value = mock_response
    return client


# ---------------------------------------------------------------------------
# TTIProvider ABC tests
# ---------------------------------------------------------------------------


def test_tti_provider_abc():
    """Test that TTIProvider is an abstract base class."""
    with pytest.raises(TypeError):
        TTIProvider()  # Can't instantiate abstract class


# ---------------------------------------------------------------------------
# CloudflareTTIProvider tests
# ---------------------------------------------------------------------------


def test_cloudflare_init_raises_value_error_when_env_vars_missing():
    """Test that missing required environment variables raises ValueError."""
    with patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": "", "CLOUDFLARE_API_TOKEN": "", "CLOUDFLARE_API_KEY": "", "GROQ_API_KEY": ""}, clear=False), \
         patch("src.modules.image.text_to_image.providers.cloudflare.settings") as mock_settings:
        mock_settings.CLOUDFLARE_ACCOUNT_ID = ""
        mock_settings.CLOUDFLARE_API_TOKEN = ""
        mock_settings.CLOUDFLARE_API_KEY = None
        mock_settings.cloudflare_account_id = ""
        mock_settings.cloudflare_api_token = ""
        mock_settings.GROQ_API_KEY = ""
        mock_settings.groq_api_key = ""

        with pytest.raises(ValueError) as exc_info:
            CloudflareTTIProvider()

        assert "Missing required environment variables" in str(exc_info.value)


def test_cloudflare_init_success_with_args():
    """Test that CloudflareTTIProvider initializes with explicit credentials."""
    with patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test"}, clear=False):
        provider = CloudflareTTIProvider(
            account_id="test_acc",
            api_token="test_tok",
            model="@cf/black-forest-labs/flux-1-schnell",
        )
        assert provider.provider_name == "cloudflare"
        assert provider.account_id == "test_acc"
        assert provider.api_token == "test_tok"
        assert provider.model == "@cf/black-forest-labs/flux-1-schnell"


@pytest.mark.asyncio
async def test_cloudflare_generate_image_empty_prompt_raises_value_error(mock_cloudflare_client):
    """Test that an empty prompt raises ValueError."""
    provider = CloudflareTTIProvider(account_id="test_acc", api_token="test_tok", client=mock_cloudflare_client)

    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        await provider.generate_image("")


@pytest.mark.asyncio
async def test_cloudflare_generate_image_success(mock_cloudflare_client):
    """Test successful image generation with Cloudflare Workers AI."""
    provider = CloudflareTTIProvider(
        account_id="test_acc",
        api_token="test_tok",
        model="@cf/black-forest-labs/flux-1-schnell",
        client=mock_cloudflare_client,
    )

    result = await provider.generate_image("A futuristic city at sunset")

    assert result == b"fake_image_bytes"
    mock_cloudflare_client.post.assert_called_once_with(
        "https://api.cloudflare.com/client/v4/accounts/test_acc/ai/run/@cf/black-forest-labs/flux-1-schnell",
        headers={"Authorization": "Bearer test_tok"},
        json={"prompt": "A futuristic city at sunset"},
        timeout=60,
    )


@pytest.mark.asyncio
async def test_cloudflare_generate_image_json_b64_response():
    """Test generating image when Cloudflare returns JSON with base64."""
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {
        "success": True,
        "result": {"image": base64.b64encode(b"decoded_image_bytes").decode("utf-8")},
    }
    client.post.return_value = mock_response

    provider = CloudflareTTIProvider(account_id="test_acc", api_token="test_tok", client=client)
    result = await provider.generate_image("A cat in sunglasses")

    assert result == b"decoded_image_bytes"


@pytest.mark.asyncio
async def test_cloudflare_generate_image_with_output_path(mock_cloudflare_client):
    """Test generating image and saving to a file."""
    provider = CloudflareTTIProvider(account_id="test_acc", api_token="test_tok", client=mock_cloudflare_client)

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_file = os.path.join(tmp_dir, "output.png")
        result = await provider.generate_image("A serene lake", output_path=output_file)

        assert result == b"fake_image_bytes"
        assert os.path.exists(output_file)
        with open(output_file, "rb") as f:
            assert f.read() == b"fake_image_bytes"


@pytest.mark.asyncio
async def test_cloudflare_generate_image_api_error_raises_tti_error():
    """Test that Cloudflare API errors are wrapped in TextToImageError."""
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.headers = {"content-type": "application/json"}
    mock_response.text = "Bad Request"
    mock_response.json.return_value = {
        "success": False,
        "errors": [{"code": 1000, "message": "Model not available"}],
    }
    client.post.return_value = mock_response

    provider = CloudflareTTIProvider(account_id="test_acc", api_token="test_tok", client=client)

    with pytest.raises(TextToImageError, match="Failed to generate image: Model not available"):
        await provider.generate_image("A cute robot")


@pytest.mark.asyncio
async def test_cloudflare_create_scenario_success():
    """Test creating a scenario from chat history on CloudflareTTIProvider."""
    provider = CloudflareTTIProvider(account_id="test_acc", api_token="test_tok", client=MagicMock())

    mock_msg = MagicMock()
    mock_msg.type = "human"
    mock_msg.content = "What are you doing now?"

    expected_scenario = ScenarioPrompt(
        narrative="I'm walking through an autumn park.",
        image_prompt="Golden leaves falling in a serene park, morning sun, 8425.HEIC",
    )

    with patch("src.modules.image.text_to_image.providers.cloudflare.ChatGroq") as mock_chat_groq:
        mock_llm_instance = MagicMock()
        mock_structured_llm = MagicMock()
        mock_chat_groq.return_value = mock_llm_instance
        mock_llm_instance.with_structured_output.return_value = mock_structured_llm

        with patch("src.modules.image.text_to_image.providers.cloudflare.PromptTemplate.__or__") as mock_pipe:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = expected_scenario
            mock_pipe.return_value = mock_chain

            result = await provider.create_scenario([mock_msg])
            assert result == expected_scenario


@pytest.mark.asyncio
async def test_cloudflare_create_scenario_failure_raises_tti_error():
    """Test error handling in create_scenario."""
    provider = CloudflareTTIProvider(account_id="test_acc", api_token="test_tok", client=MagicMock())

    with patch("src.modules.image.text_to_image.providers.cloudflare.ChatGroq", side_effect=RuntimeError("Groq offline")):
        with pytest.raises(TextToImageError, match="Failed to create scenario: Groq offline"):
            await provider.create_scenario([])


@pytest.mark.asyncio
async def test_cloudflare_enhance_prompt_success():
    """Test enhancing prompt on CloudflareTTIProvider."""
    provider = CloudflareTTIProvider(account_id="test_acc", api_token="test_tok", client=MagicMock())

    expected_enhanced = EnhancedPrompt(
        content="photo of a person having coffee in a cozy cafe, morning light, 8425.HEIC"
    )

    with patch("src.modules.image.text_to_image.providers.cloudflare.ChatGroq") as mock_chat_groq:
        mock_llm_instance = MagicMock()
        mock_structured_llm = MagicMock()
        mock_chat_groq.return_value = mock_llm_instance
        mock_llm_instance.with_structured_output.return_value = mock_structured_llm

        with patch("src.modules.image.text_to_image.providers.cloudflare.PromptTemplate.__or__") as mock_pipe:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = expected_enhanced
            mock_pipe.return_value = mock_chain

            result = await provider.enhance_prompt("person having coffee")
            assert result == expected_enhanced.content


@pytest.mark.asyncio
async def test_cloudflare_enhance_prompt_failure_raises_tti_error():
    """Test error handling in enhance_prompt."""
    provider = CloudflareTTIProvider(account_id="test_acc", api_token="test_tok", client=MagicMock())

    with patch("src.modules.image.text_to_image.providers.cloudflare.ChatGroq", side_effect=RuntimeError("Groq error")):
        with pytest.raises(TextToImageError, match="Failed to enhance prompt: Groq error"):
            await provider.enhance_prompt("simple prompt")


# ---------------------------------------------------------------------------
# TTIFactory tests
# ---------------------------------------------------------------------------


def test_tti_factory_creates_cloudflare_by_default():
    """Test that TTIFactory.create() defaults to Cloudflare based on settings.TTI_PROVIDER."""
    with patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": "acc", "CLOUDFLARE_API_TOKEN": "tok", "GROQ_API_KEY": "groq"}, clear=False):
        provider = TTIFactory.create(force_new=True)
        assert isinstance(provider, CloudflareTTIProvider)
        assert provider.provider_name == "cloudflare"


def test_tti_factory_create_with_enum():
    """Test that TTIFactory creates provider using enum."""
    with patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": "acc", "CLOUDFLARE_API_TOKEN": "tok", "GROQ_API_KEY": "groq"}, clear=False):
        provider = TTIFactory.create(TTIProviderType.CLOUDFLARE, force_new=True)
        assert isinstance(provider, CloudflareTTIProvider)


def test_tti_factory_unsupported_provider_raises_value_error():
    """Test that requesting an unsupported provider raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported TTI provider: 'invalid_provider'"):
        TTIFactory.create("invalid_provider")


def test_tti_factory_register_custom_provider():
    """Test registering a custom TTI provider in TTIFactory."""
    class CustomProvider(TTIProvider):
        @property
        def provider_name(self) -> str:
            return "custom"

        async def generate_image(self, prompt: str, output_path: str = "") -> bytes:
            return b"custom_bytes"

        async def create_scenario(self, chat_history: list = None) -> ScenarioPrompt:
            return ScenarioPrompt(narrative="test", image_prompt="test")

        async def enhance_prompt(self, prompt: str) -> str:
            return "enhanced"

    TTIFactory.register_provider("custom", lambda s: CustomProvider())
    custom = TTIFactory.create("custom", force_new=True)
    assert isinstance(custom, CustomProvider)
    assert custom.provider_name == "custom"


def test_get_tti_provider_convenience():
    """Test get_tti_provider helper function."""
    with patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": "acc", "CLOUDFLARE_API_TOKEN": "tok", "GROQ_API_KEY": "groq"}, clear=False):
        provider = get_tti_provider()
        assert isinstance(provider, CloudflareTTIProvider)


# ---------------------------------------------------------------------------
# TextToImage integration & delegation tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_to_image_delegates_to_provider():
    """Test that TextToImage delegates generate_image, create_scenario, and enhance_prompt to provider."""
    mock_provider = MagicMock(spec=TTIProvider)
    mock_provider.generate_image.return_value = b"delegated_bytes"
    mock_provider.create_scenario.return_value = ScenarioPrompt(narrative="n", image_prompt="p")
    mock_provider.enhance_prompt.return_value = "enhanced_text"

    tti = TextToImage(provider=mock_provider)

    img = await tti.generate_image("A prompt", output_path="out.png")
    assert img == b"delegated_bytes"
    mock_provider.generate_image.assert_called_once_with(prompt="A prompt", output_path="out.png")

    scenario = await tti.create_scenario([])
    assert scenario.narrative == "n"
    mock_provider.create_scenario.assert_called_once_with(chat_history=[])

    enhanced = await tti.enhance_prompt("simple")
    assert enhanced == "enhanced_text"
    mock_provider.enhance_prompt.assert_called_once_with(prompt="simple")


def test_get_text_to_image_singleton():
    """Test singleton pattern in get_text_to_image()."""
    mock_provider = MagicMock(spec=TTIProvider)
    inst1 = get_text_to_image(provider=mock_provider)
    inst2 = get_text_to_image()
    assert inst1 is inst2


def test_settings_properties():
    """Test that Cloudflare and TTI settings properties exist."""
    assert hasattr(settings, "TTI_PROVIDER")
    assert settings.TTI_PROVIDER == "cloudflare"
    assert hasattr(settings, "CLOUDFLARE_ACCOUNT_ID")
    assert hasattr(settings, "CLOUDFLARE_API_TOKEN")
    assert hasattr(settings, "TTI_MODEL_NAME")
    assert hasattr(settings, "TEXT_MODEL_NAME")
    assert settings.TTI_MODEL_NAME == "@cf/black-forest-labs/flux-1-schnell"
    assert settings.TEXT_MODEL_NAME == "llama-3.3-70b-versatile"
    assert hasattr(settings, "cloudflare")
    assert settings.cloudflare.model == "@cf/black-forest-labs/flux-1-schnell"
    # Ensure Together-specific properties are removed
    assert not hasattr(settings, "TOGETHER_API_KEY")
