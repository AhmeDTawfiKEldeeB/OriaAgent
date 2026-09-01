import asyncio
import os
import sys

# Ensure project root is in sys.path when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.settings import get_settings
from src.infrastructure.llm import get_llm_provider


def test_active_provider():
    """Test the active provider configured in .env."""
    settings = get_settings()
    provider_name = settings.llm_provider.value

    print(f"\n==========================================")
    print(f" Testing Active Provider: {provider_name.upper()}")
    print(f"==========================================")

    # 1. Initialize Provider via Factory
    provider = get_llm_provider()
    print(f"[1/4] Provider Initialized: {provider.provider_name} (Model: {provider.model_name})")
    assert provider.provider_name == provider_name

    # 2. Test Synchronous Generation
    print(f"[2/4] Testing Sync Generation...")
    response = provider.generate("Reply with 'Hello' and nothing else.")
    print(f"      Response: {response.strip()}")
    assert len(response.strip()) > 0

    # 3. Test Streaming
    print(f"[3/4] Testing Streaming...")
    chunks = []
    print("      Stream output: ", end="", flush=True)
    for chunk in provider.stream("Count 1, 2, 3"):
        print(chunk, end="", flush=True)
        chunks.append(chunk)
    print()
    assert len(chunks) > 0

    # 4. Test Async Generation
    print(f"[4/4] Testing Async Generation...")

    async def run_async():
        res = await provider.agenerate("Say 'Async OK'.")
        print(f"      Async Response: {res.strip()}")
        assert len(res.strip()) > 0

    asyncio.run(run_async())

    print(f"\nSUCCESS: Provider '{provider_name}' passed all tests!\n")


if __name__ == "__main__":
    test_active_provider()
