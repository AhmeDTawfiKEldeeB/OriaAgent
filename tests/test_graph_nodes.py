from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.nodes import (
    audio_node,
    context_injection_node,
    conversation_node,
    image_node,
    memory_extraction_node,
    memory_injection_node,
    router_node,
    summarize_conversation_node,
)
from src.graph.state import OriaState


@pytest.mark.asyncio
async def test_router_node():
    state: OriaState = {
        "messages": [HumanMessage(content="Send me an audio")],
        "summary": "",
        "workflow": "",
        "audio_buffer": b"",
        "image_path": "",
        "current_activity": "",
        "apply_activity": False,
        "memory_context": "",
    }
    with patch("src.graph.nodes.get_router_chain") as mock_chain_fn:
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value=MagicMock(response_type="audio"))
        mock_chain_fn.return_value = mock_chain

        res = await router_node(state)
        assert res == {"workflow": "audio"}


def test_context_injection_node():
    state: OriaState = {
        "messages": [],
        "summary": "",
        "workflow": "",
        "audio_buffer": b"",
        "image_path": "",
        "current_activity": "old activity",
        "apply_activity": False,
        "memory_context": "",
    }
    with patch(
        "src.graph.nodes.ScheduleContextGenerator.get_current_activity",
        return_value="new activity",
    ):
        res = context_injection_node(state)
        assert res["apply_activity"] is True
        assert res["current_activity"] == "new activity"


@pytest.mark.asyncio
async def test_conversation_node():
    state: OriaState = {
        "messages": [HumanMessage(content="Hi Oria")],
        "summary": "Previous summary",
        "workflow": "conversation",
        "audio_buffer": b"",
        "image_path": "",
        "current_activity": "coding",
        "apply_activity": False,
        "memory_context": "likes coffee",
    }
    with patch("src.graph.nodes.get_character_response_chain") as mock_chain_fn:
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value="Hey! What's up?")
        mock_chain_fn.return_value = mock_chain

        res = await conversation_node(state, config={})
        assert isinstance(res["messages"], AIMessage)
        assert res["messages"].content == "Hey! What's up?"


@pytest.mark.asyncio
async def test_audio_node():
    state: OriaState = {
        "messages": [HumanMessage(content="Say hi")],
        "summary": "",
        "workflow": "audio",
        "audio_buffer": b"",
        "image_path": "",
        "current_activity": "",
        "apply_activity": False,
        "memory_context": "",
    }
    with (
        patch("src.graph.nodes.get_character_response_chain") as mock_chain_fn,
        patch("src.graph.nodes.get_text_to_speech_module") as mock_tts_fn,
    ):
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value="Hi there!")
        mock_chain_fn.return_value = mock_chain

        mock_tts = MagicMock()
        mock_tts.synthesize = AsyncMock(return_value=b"audio_bytes")
        mock_tts_fn.return_value = mock_tts

        res = await audio_node(state, config={})
        assert isinstance(res["messages"], AIMessage)
        assert res["messages"].content == "Hi there!"
        assert res["audio_buffer"] == b"audio_bytes"


def test_memory_injection_node():
    state: OriaState = {
        "messages": [HumanMessage(content="I love hiking")],
        "summary": "",
        "workflow": "",
        "audio_buffer": b"",
        "image_path": "",
        "current_activity": "",
        "apply_activity": False,
        "memory_context": "",
    }
    with patch("src.graph.nodes.get_memory_manager") as mock_mm_fn:
        mock_mm = MagicMock()
        mock_mm.get_relevant_memories.return_value = ["Loves hiking"]
        mock_mm.format_memories_for_prompt.return_value = "- Loves hiking"
        mock_mm_fn.return_value = mock_mm

        res = memory_injection_node(state)
        assert res == {"memory_context": "- Loves hiking"}
