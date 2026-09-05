import argparse
import asyncio
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.settings import get_settings
from src.core.exceptions import TextToImageError
from src.modules.image.text_to_image import get_text_to_image


async def test_real_tti(
    prompt: str = "a cat wearing a spacesuit, floating in space, with Earth in the background, highly detailed, cinematic lighting",
    output_filename: str = "test_generated_image.png",
    model: str | None = None,
) -> bool:
    """Run real text-to-image generation using configured Cloudflare Workers AI credentials."""
    print("=" * 65)
    print(" Cloudflare Text-To-Image (TTI) Real Test")
    print("=" * 65)

    settings = get_settings()

    # Check credentials
    account_id = settings.CLOUDFLARE_ACCOUNT_ID or os.getenv("CLOUDFLARE_ACCOUNT_ID")
    api_token = (
        settings.CLOUDFLARE_API_TOKEN
        or os.getenv("CLOUDFLARE_API_TOKEN")
        or os.getenv("CLOUDFLARE_API_KEY")
    )
    model_name = model or settings.TTI_MODEL_NAME or "@cf/black-forest-labs/flux-1-schnell"

    if not account_id or not api_token:
        print("\n[!] ERROR: Missing Cloudflare credentials.")
        print("    Please ensure CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN are set in your .env file.")
        print(f"    CLOUDFLARE_ACCOUNT_ID: {'[SET]' if account_id else '[MISSING]'}")
        print(f"    CLOUDFLARE_API_TOKEN : {'[SET]' if api_token else '[MISSING]'}")
        print("\n[TIP] Add the following to your .env file:")
        print("      CLOUDFLARE_ACCOUNT_ID=your_account_id")
        print("      CLOUDFLARE_API_TOKEN=your_api_token")
        print(f"      TTI_MODEL_NAME={model_name}")
        return False

    masked_token = f"{api_token[:6]}...{api_token[-4:]}" if len(api_token) > 10 else "***"
    print(f"\n[1/3] Initializing TTI provider...")
    print(f"      Account ID: {account_id}")
    print(f"      API Token : {masked_token}")
    print(f"      Model     : {model_name}")

    try:
        tti = get_text_to_image()
        print("      Provider successfully initialized.")
    except Exception as e:
        print(f"\n[!] Failed to initialize TextToImage: {e}")
        return False

    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", output_filename))
    print(f"\n[2/3] Generating image from query...")
    print(f"      Prompt: \"{prompt}\"")

    try:
        image_bytes = await tti.generate_image(prompt=prompt, output_path=output_path)
        size_kb = len(image_bytes) / 1024
        print(f"      Generated image size: {len(image_bytes):,} bytes ({size_kb:.1f} KB)")
    except TextToImageError as e:
        print(f"\n[!] Image generation failed: {e}")
        return False
    except Exception as e:
        print(f"\n[!] Unexpected error during generation: {e}")
        return False

    if not image_bytes or not os.path.exists(output_path):
        print("\n[!] No image file was saved.")
        return False

    print(f"\n[3/3] Verifying output file...")
    print(f"      Image saved successfully to: {output_path}")

    print("\n" + "=" * 65)
    print(" SUCCESS: Text-to-image generation completed successfully!")
    print(f" Photo saved at: {output_path}")
    print("=" * 65)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Cloudflare Text-to-Image generation.")
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Text query/prompt describing the photo to generate",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="test_generated_image.png",
        help="Output image filename (default: test_generated_image.png)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Cloudflare TTI model name (optional)",
    )
    args = parser.parse_args()

    call_kwargs = {}
    if args.query is not None:
        call_kwargs["prompt"] = args.query
    if args.output is not None:
        call_kwargs["output_filename"] = args.output
    if args.model is not None:
        call_kwargs["model"] = args.model

    asyncio.run(test_real_tti(**call_kwargs))
