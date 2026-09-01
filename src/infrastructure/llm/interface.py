from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Iterator, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage


class BaseLLMProvider(ABC):
    """Abstract Base Class for LLM Providers leveraging LangChain ChatModels."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the LLM provider."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier used by the provider."""
        pass

    @property
    @abstractmethod
    def chat_model(self) -> BaseChatModel:
        """Underlying LangChain BaseChatModel instance."""
        pass

    def generate(
        self,
        input_data: Union[str, list[BaseMessage], list[dict[str, Any]]],
        **kwargs: Any,
    ) -> str:
        """Synchronously generate response text using LangChain invoke."""
        response = self.chat_model.invoke(input_data, **kwargs)
        return str(response.content) if response.content is not None else ""

    async def agenerate(
        self,
        input_data: Union[str, list[BaseMessage], list[dict[str, Any]]],
        **kwargs: Any,
    ) -> str:
        """Asynchronously generate response text using LangChain ainvoke."""
        response = await self.chat_model.ainvoke(input_data, **kwargs)
        return str(response.content) if response.content is not None else ""

    def stream(
        self,
        input_data: Union[str, list[BaseMessage], list[dict[str, Any]]],
        **kwargs: Any,
    ) -> Iterator[str]:
        """Synchronously stream response chunks using LangChain stream."""
        for chunk in self.chat_model.stream(input_data, **kwargs):
            if chunk.content:
                yield str(chunk.content)

    async def astream(
        self,
        input_data: Union[str, list[BaseMessage], list[dict[str, Any]]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Asynchronously stream response chunks using LangChain astream."""
        async for chunk in self.chat_model.astream(input_data, **kwargs):
            if chunk.content:
                yield str(chunk.content)
