from src.config.settings import get_settings
from src.infrastructure.llm import get_chat_model, get_llm_provider


def main():
    settings = get_settings()
    print(f"Loaded Environment: {settings.app_env}")
    print(f"Active LLM Provider: {settings.llm_provider.value}")

    # Initialize provider from current .env settings
    try:
        provider = get_llm_provider()
        print(f"Successfully initialized provider: {provider.provider_name} (Model: {provider.model_name})")
        print(f"Underlying LangChain Chat Model: {type(provider.chat_model).__name__}")
    except ValueError as e:
        print(f"Provider initialization notice: {e}")


if __name__ == "__main__":
    main()
