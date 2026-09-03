import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.settings import MemoryProviderType, Settings
from src.modules.memory.long_term.interface import BaseVectorStore, Memory
from src.modules.memory.memory_manager import (
    MemoryAnalysis,
    MemoryManager,
    get_memory_manager,
)


@pytest.fixture
def mock_vector_store():
    """Mock vector store implementing BaseVectorStore."""
    store = MagicMock(spec=BaseVectorStore)
    store.provider_name = "mock_store"
    store.find_similar_memory.return_value = None
    store.store_memory.return_value = "mem-12345"
    store.search_memories.return_value = [
        Memory(text="User likes Python", score=0.95),
        Memory(text="User prefers MongoDB", score=0.88),
    ]
    return store


@pytest.fixture
def mock_llm():
    """Mock structured LLM that returns a MemoryAnalysis instance."""
    llm = AsyncMock()
    llm.ainvoke.return_value = MemoryAnalysis(
        is_important=True,
        formatted_memory="User likes Python programming",
    )
    return llm


def test_memory_analysis_model():
    """Test MemoryAnalysis pydantic model serialization and validation."""
    analysis = MemoryAnalysis(
        is_important=True,
        formatted_memory="User lives in Cairo",
    )
    assert analysis.is_important is True
    assert analysis.formatted_memory == "User lives in Cairo"

    empty_analysis = MemoryAnalysis(is_important=False)
    assert empty_analysis.is_important is False
    assert empty_analysis.formatted_memory is None


def test_memory_manager_init_with_custom_store(mock_vector_store, mock_llm):
    """Test initializing MemoryManager with an injected vector store and LLM."""
    manager = MemoryManager(
        vector_store=mock_vector_store,
        llm=mock_llm,
    )
    assert manager.vector_store == mock_vector_store
    assert manager.llm == mock_llm


def test_memory_manager_init_qdrant_provider():
    """Test initializing MemoryManager explicitly selecting Qdrant."""
    with patch("src.modules.memory.long_term.factory.MemoryFactory.create") as mock_create:
        mock_store = MagicMock(spec=BaseVectorStore)
        mock_store.provider_name = "qdrant"
        mock_create.return_value = mock_store

        mock_llm = AsyncMock()
        manager = MemoryManager(provider=MemoryProviderType.QDRANT, llm=mock_llm)

        mock_create.assert_called_once()
        assert manager.vector_store.provider_name == "qdrant"


def test_memory_manager_init_mongodb_provider():
    """Test initializing MemoryManager explicitly selecting MongoDB."""
    with patch("src.modules.memory.long_term.factory.MemoryFactory.create") as mock_create:
        mock_store = MagicMock(spec=BaseVectorStore)
        mock_store.provider_name = "mongodb"
        mock_create.return_value = mock_store

        mock_llm = AsyncMock()
        manager = MemoryManager(provider="mongodb", llm=mock_llm)

        mock_create.assert_called_once()
        assert manager.vector_store.provider_name == "mongodb"


@pytest.mark.asyncio
async def test_extract_and_store_memories_human_message_success(mock_vector_store, mock_llm):
    """Test storing memory from a human message when important and no duplicate exists."""
    manager = MemoryManager(vector_store=mock_vector_store, llm=mock_llm)

    human_msg = HumanMessage(content="I prefer building backend systems with FastAPI and Python.")
    mem_id = await manager.extract_and_store_memories(human_msg)

    assert mem_id == "mem-12345"
    mock_llm.ainvoke.assert_awaited_once()
    mock_vector_store.find_similar_memory.assert_called_once_with("User likes Python programming")
    mock_vector_store.store_memory.assert_called_once()
    args, kwargs = mock_vector_store.store_memory.call_args
    assert kwargs["text"] == "User likes Python programming"
    assert "timestamp" in kwargs["metadata"]
    assert "id" in kwargs["metadata"]


@pytest.mark.asyncio
async def test_extract_and_store_memories_skips_non_human_messages(mock_vector_store, mock_llm):
    """Test that non-human messages (AI, System) are ignored for memory storage."""
    manager = MemoryManager(vector_store=mock_vector_store, llm=mock_llm)

    ai_msg = AIMessage(content="I can help you with Python and FastAPI!")
    res1 = await manager.extract_and_store_memories(ai_msg)
    assert res1 is None

    sys_msg = SystemMessage(content="You are a helpful assistant.")
    res2 = await manager.extract_and_store_memories(sys_msg)
    assert res2 is None

    mock_llm.ainvoke.assert_not_called()
    mock_vector_store.store_memory.assert_not_called()


@pytest.mark.asyncio
async def test_extract_and_store_memories_skips_duplicate(mock_vector_store, mock_llm):
    """Test that memory storage is skipped when a similar memory already exists."""
    # Existing similar memory returned by store
    mock_vector_store.find_similar_memory.return_value = Memory(
        text="User likes Python programming",
        score=0.95,
    )
    manager = MemoryManager(vector_store=mock_vector_store, llm=mock_llm)

    human_msg = HumanMessage(content="I really enjoy Python programming!")
    res = await manager.extract_and_store_memories(human_msg)

    assert res is None
    mock_vector_store.find_similar_memory.assert_called_once()
    mock_vector_store.store_memory.assert_not_called()


@pytest.mark.asyncio
async def test_extract_and_store_memories_skips_unimportant(mock_vector_store):
    """Test that casual chat / unimportant messages are not stored."""
    unimportant_llm = AsyncMock()
    unimportant_llm.ainvoke.return_value = MemoryAnalysis(
        is_important=False,
        formatted_memory=None,
    )

    manager = MemoryManager(vector_store=mock_vector_store, llm=unimportant_llm)

    human_msg = HumanMessage(content="Hello! How are you doing today?")
    res = await manager.extract_and_store_memories(human_msg)

    assert res is None
    mock_vector_store.find_similar_memory.assert_not_called()
    mock_vector_store.store_memory.assert_not_called()


def test_get_relevant_memories(mock_vector_store):
    """Test retrieving relevant memories as text strings."""
    manager = MemoryManager(vector_store=mock_vector_store, llm=MagicMock())

    memories = manager.get_relevant_memories("Tell me about Python", k=2)

    assert len(memories) == 2
    assert memories == ["User likes Python", "User prefers MongoDB"]
    mock_vector_store.search_memories.assert_called_once_with("Tell me about Python", k=2)


def test_format_memories_for_prompt(mock_vector_store):
    """Test formatting retrieved memories as markdown bullets."""
    manager = MemoryManager(vector_store=mock_vector_store, llm=MagicMock())

    formatted = manager.format_memories_for_prompt(["User likes Python", "User prefers MongoDB"])
    assert formatted == "- User likes Python\n- User prefers MongoDB"

    empty = manager.format_memories_for_prompt([])
    assert empty == ""


def test_get_memory_manager_helper(mock_vector_store, mock_llm):
    """Test get_memory_manager factory function."""
    mgr = get_memory_manager(vector_store=mock_vector_store, llm=mock_llm)
    assert isinstance(mgr, MemoryManager)
    assert mgr.vector_store == mock_vector_store
