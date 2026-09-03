import logging
import uuid
from datetime import datetime
from typing import Any, List, Optional, Union

from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from src.config.settings import MemoryProviderType, Settings, get_settings, settings
from src.core.prompts import MEMORY_ANALYSIS_PROMPT
from src.modules.memory.long_term.factory import get_vector_store
from src.modules.memory.long_term.interface import BaseVectorStore


class MemoryAnalysis(BaseModel):
    """Result of analyzing a message for memory-worthy content."""

    is_important: bool = Field(
        ...,
        description="Whether the message is important enough to be stored as a memory",
    )
    formatted_memory: Optional[str] = Field(
        default=None,
        description="The formatted memory to be stored",
    )


class MemoryManager:
    """Manager class for handling long-term memory operations.

    Supports both Qdrant and MongoDB vector store backends seamlessly.
    """

    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        provider: Optional[Union[MemoryProviderType, str]] = None,
        settings: Optional[Settings] = None,
        llm: Optional[Any] = None,
    ):
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(__name__)

        # 1. Initialize Vector Store (supports 'qdrant', 'mongodb', or active provider from settings)
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            self.vector_store = get_vector_store(provider=provider, settings=self.settings)

        # 2. Initialize LLM (injected or default ChatGroq with fallback)
        if llm is not None:
            self.llm = llm
        else:
            self.llm = self._init_default_llm()

    def _init_default_llm(self) -> Any:
        """Initialize the default LLM for structured memory analysis."""
        api_key = self.settings.groq_api_key or getattr(self.settings, "GROQ_API_KEY", "")
        model_name = (
            getattr(self.settings, "small_text_model_name", None)
            or getattr(self.settings, "SMALL_TEXT_MODEL_NAME", None)
            or self.settings.groq_model
        )

        if api_key:
            return ChatGroq(
                model=model_name,
                api_key=api_key,
                temperature=0.1,
                max_retries=2,
            ).with_structured_output(MemoryAnalysis)

        # Fallback to Gemini if Gemini API key is configured
        if self.settings.gemini_api_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                return ChatGoogleGenerativeAI(
                    model=self.settings.gemini_model,
                    api_key=self.settings.gemini_api_key,
                    temperature=0.1,
                ).with_structured_output(MemoryAnalysis)
            except Exception as e:
                self.logger.warning(f"Failed to initialize Gemini fallback: {e}")

        # Non-blocking fallback for offline/test environments
        return ChatGroq(
            model=model_name,
            api_key=api_key or "gsk_placeholder_key",
            temperature=0.1,
            max_retries=2,
        ).with_structured_output(MemoryAnalysis)

    async def _analyze_memory(self, message: str) -> MemoryAnalysis:
        """Analyze a message to determine importance and format if needed."""
        prompt = MEMORY_ANALYSIS_PROMPT.format(message=message)
        return await self.llm.ainvoke(prompt)

    async def extract_and_store_memories(self, message: Union[BaseMessage, Any]) -> Optional[str]:
        """Extract important information from a message and store in vector store."""
        if not hasattr(message, "type") or message.type != "human":
            return None

        content = getattr(message, "content", "")
        if isinstance(content, list):
            text_parts = [
                part if isinstance(part, str) else str(part.get("text", "")) for part in content
            ]
            content = " ".join(text_parts)
        else:
            content = str(content)

        if not content.strip():
            return None

        # Analyze the message for importance and formatting
        analysis = await self._analyze_memory(content)
        if analysis.is_important and analysis.formatted_memory:
            # Check if similar memory exists
            similar = self.vector_store.find_similar_memory(analysis.formatted_memory)
            if similar:
                # Skip storage if we already have a similar memory
                self.logger.info(f"Similar memory already exists: '{analysis.formatted_memory}'")
                return None

            # Store new memory
            self.logger.info(f"Storing new memory: '{analysis.formatted_memory}'")
            memory_id = self.vector_store.store_memory(
                text=analysis.formatted_memory,
                metadata={
                    "id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat(),
                },
            )
            return memory_id

        return None

    def get_relevant_memories(self, context: str, k: Optional[int] = None) -> List[str]:
        """Retrieve relevant memories based on the current context."""
        top_k = (
            k
            if k is not None
            else getattr(
                self.settings,
                "MEMORY_TOP_K",
                getattr(self.settings, "memory_top_k", 5),
            )
        )
        memories = self.vector_store.search_memories(context, k=top_k)
        if memories:
            for memory in memories:
                score_str = f"{memory.score:.2f}" if memory.score is not None else "N/A"
                self.logger.debug(f"Memory: '{memory.text}' (score: {score_str})")
        return [memory.text for memory in memories]

    def format_memories_for_prompt(self, memories: List[str]) -> str:
        """Format retrieved memories as bullet points."""
        if not memories:
            return ""
        return "\n".join(f"- {memory}" for memory in memories)


def get_memory_manager(
    vector_store: Optional[BaseVectorStore] = None,
    provider: Optional[Union[MemoryProviderType, str]] = None,
    settings: Optional[Settings] = None,
    llm: Optional[Any] = None,
) -> MemoryManager:
    """Get a MemoryManager instance supporting Qdrant and MongoDB."""
    return MemoryManager(
        vector_store=vector_store,
        provider=provider,
        settings=settings,
        llm=llm,
    )
