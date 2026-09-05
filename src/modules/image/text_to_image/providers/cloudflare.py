import base64
import logging
import os
from typing import Any, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

from src.config.settings import settings
from src.core.exceptions import TextToImageError
from src.core.prompts import IMAGE_ENHANCEMENT_PROMPT, IMAGE_SCENARIO_PROMPT
from src.modules.image.text_to_image.tti_interface import (
    EnhancedPrompt,
    ScenarioPrompt,
    TTIProvider,
)


from langchain_core.prompts import PromptTemplate

from langchain_groq import ChatGroq


class CloudflareTTIProvider(TTIProvider):
    """Text-to-image provider using Cloudflare Workers AI."""

    REQUIRED_ENV_VARS = ["GROQ_API_KEY", "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"]

    def __init__(
        self,
        account_id: Optional[str] = None,
        api_token: Optional[str] = None,
        model: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        """Initialize CloudflareTTIProvider and validate environment variables."""
        self._account_id = account_id
        self._api_token = api_token
        self._model = model
        self._client = client
        self.logger = logging.getLogger(__name__)

        if not self._client and not (self._account_id and self._api_token):
            self._validate_env_vars()

    def _validate_env_vars(self) -> None:
        """Validate that all required environment variables are set."""
        missing_vars = []
        for var in self.REQUIRED_ENV_VARS:
            val = (
                (self._account_id if var == "CLOUDFLARE_ACCOUNT_ID" else None)
                or (self._api_token if var == "CLOUDFLARE_API_TOKEN" else None)
                or os.getenv(var)
                or getattr(settings, var, None)
                or getattr(settings, var.lower(), None)
            )
            # Support CLOUDFLARE_API_KEY as an alternative to CLOUDFLARE_API_TOKEN
            if not val and var == "CLOUDFLARE_API_TOKEN":
                val = os.getenv("CLOUDFLARE_API_KEY") or getattr(settings, "CLOUDFLARE_API_KEY", None)

            if not val:
                missing_vars.append(var)

        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    @property
    def provider_name(self) -> str:
        """Name of the provider."""
        return "cloudflare"

    @property
    def account_id(self) -> str:
        """Get Cloudflare Account ID."""
        return (
            self._account_id
            or getattr(settings, "CLOUDFLARE_ACCOUNT_ID", "")
            or os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
        )

    @property
    def api_token(self) -> str:
        """Get Cloudflare API Token."""
        return (
            self._api_token
            or getattr(settings, "CLOUDFLARE_API_TOKEN", "")
            or os.getenv("CLOUDFLARE_API_TOKEN", "")
            or getattr(settings, "CLOUDFLARE_API_KEY", "")
            or os.getenv("CLOUDFLARE_API_KEY", "")
        )

    @property
    def model(self) -> str:
        """Get Cloudflare text-to-image model identifier."""
        return (
            self._model
            or getattr(settings, "TTI_MODEL_NAME", "")
            or "@cf/black-forest-labs/flux-1-schnell"
        )

    async def generate_image(self, prompt: str, output_path: str = "") -> bytes:
        """Generate an image from a prompt using Cloudflare Workers AI."""
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        try:
            self.logger.info(f"Generating image for prompt: '{prompt}'")

            url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.model}"
            headers = {
                "Authorization": f"Bearer {self.api_token}",
            }
            payload = {"prompt": prompt}

            client = self._client or requests
            response = client.post(url, headers=headers, json=payload, timeout=60)

            if response.status_code != 200:
                error_msg = f"Cloudflare API returned status {response.status_code}: {response.text}"
                try:
                    err_json = response.json()
                    errors = err_json.get("errors", [])
                    if errors:
                        error_msg = ", ".join([e.get("message", str(e)) for e in errors])
                except Exception:
                    pass
                raise TextToImageError(f"Failed to generate image: {error_msg}")

            content_type = response.headers.get("content-type", "") if hasattr(response, "headers") else ""
            if "application/json" in content_type:
                data = response.json()
                if not data.get("success", True):
                    errors = data.get("errors", [])
                    err_detail = ", ".join([e.get("message", str(e)) for e in errors]) or "Unknown Cloudflare error"
                    raise TextToImageError(f"Failed to generate image: {err_detail}")
                result = data.get("result", {})
                if isinstance(result, dict) and "image" in result:
                    image_data = base64.b64decode(result["image"])
                else:
                    image_data = response.content
            else:
                image_data = response.content

            if not image_data:
                raise TextToImageError("Failed to generate image: Empty response from Cloudflare API")

            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(image_data)
                self.logger.info(f"Image saved to {output_path}")

            return image_data

        except Exception as e:
            if isinstance(e, (ValueError, TextToImageError)):
                raise
            raise TextToImageError(f"Failed to generate image: {str(e)}") from e

    async def create_scenario(self, chat_history: list = None) -> ScenarioPrompt:
        """Creates a first-person narrative scenario and corresponding image prompt based on chat history."""
        try:
            if chat_history is None:
                chat_history = []
            formatted_history = "\n".join([
                f"{getattr(msg, 'type', type(msg).__name__).title()}: {getattr(msg, 'content', str(msg))}"
                for msg in chat_history[-5:]
            ])

            self.logger.info("Creating scenario from chat history")

            llm = ChatGroq(
                model=settings.TEXT_MODEL_NAME,
                api_key=settings.GROQ_API_KEY,
                temperature=0.4,
                max_retries=2,
            )

            structured_llm = llm.with_structured_output(ScenarioPrompt)

            chain = (
                PromptTemplate(
                    input_variables=["chat_history"],
                    template=IMAGE_SCENARIO_PROMPT,
                )
                | structured_llm
            )

            scenario = chain.invoke({"chat_history": formatted_history})
            self.logger.info(f"Created scenario: {scenario}")

            return scenario

        except Exception as e:
            raise TextToImageError(f"Failed to create scenario: {str(e)}") from e

    async def enhance_prompt(self, prompt: str) -> str:
        """Enhance a simple prompt with additional details and context."""
        try:
            self.logger.info(f"Enhancing prompt: '{prompt}'")

            llm = ChatGroq(
                model=settings.TEXT_MODEL_NAME,
                api_key=settings.GROQ_API_KEY,
                temperature=0.25,
                max_retries=2,
            )

            structured_llm = llm.with_structured_output(EnhancedPrompt)

            chain = (
                PromptTemplate(
                    input_variables=["prompt"],
                    template=IMAGE_ENHANCEMENT_PROMPT,
                )
                | structured_llm
            )

            enhanced_prompt = chain.invoke({"prompt": prompt}).content
            self.logger.info(f"Enhanced prompt: '{enhanced_prompt}'")

            return enhanced_prompt

        except Exception as e:
            raise TextToImageError(f"Failed to enhance prompt: {str(e)}") from e
