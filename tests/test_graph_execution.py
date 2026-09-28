from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.config.settings import settings
from src.graph.graph import graph
from src.graph.utils.chains import RouterResponse


@pytest.mark.asyncio
async def test_graph_conversation_execution_mocked():
    """Test full graph execution for normal conversation workflow without external API calls."""
    with (
        patch("src.graph.nodes.get_router_chain") as mock_router_fn,
        patch("src.graph.nodes.get_character_response_chain") as mock_char_fn,
        patch("src.graph.nodes.get_memory_manager") as mock_mm_fn,
    ):
        # 1. Mock Router
        mock_router_chain = MagicMock()
        mock_router_chain.ainvoke = AsyncMock(
            return_value=RouterResponse(response_type="conversation")
        )
        mock_router_fn.return_value = mock_router_chain

        # 2. Mock Character Response Chain
        mock_char_chain = MagicMock()
        mock_char_chain.ainvoke = AsyncMock(return_value="Hey! What's up?")
        mock_char_fn.return_value = mock_char_chain

        # 3. Mock Memory Manager
        mock_mm = MagicMock()
        mock_mm.get_relevant_memories.return_value = ["User likes coding"]
        mock_mm.format_memories_for_prompt.return_value = "- User likes coding"
        mock_mm.extract_and_store_memories = AsyncMock(return_value=None)
        mock_mm_fn.return_value = mock_mm

        # Execute Graph
        input_state = {"messages": [HumanMessage(content="Hello Oria!")]}
        result = await graph.ainvoke(input_state)

        # Assertions
        assert "messages" in result
        assert len(result["messages"]) >= 2
        last_message = result["messages"][-1]
        assert isinstance(last_message, AIMessage)
        assert last_message.content == "Hey! What's up?"
        assert result.get("workflow") == "conversation"
        assert result.get("current_activity") != ""
        assert "- User likes coding" in result.get("memory_context", "")


@pytest.mark.asyncio
async def test_graph_image_execution_mocked(tmp_path):
    """Test full graph execution for image workflow."""
    with (
        patch("src.graph.nodes.get_router_chain") as mock_router_fn,
        patch("src.graph.nodes.get_character_response_chain") as mock_char_fn,
        patch("src.graph.nodes.get_text_to_image_module") as mock_tti_fn,
        patch("src.graph.nodes.get_memory_manager") as mock_mm_fn,
    ):
        # 1. Mock Router to select image
        mock_router_chain = MagicMock()
        mock_router_chain.ainvoke = AsyncMock(
            return_value=RouterResponse(response_type="image")
        )
        mock_router_fn.return_value = mock_router_chain

        # 2. Mock TTI Module
        mock_tti = MagicMock()
        mock_scenario = MagicMock()
        mock_scenario.image_prompt = "A sunset over the bay"
        mock_tti.create_scenario = AsyncMock(return_value=mock_scenario)
        mock_tti.generate_image = AsyncMock(return_value=b"fake_image_bytes")
        mock_tti_fn.return_value = mock_tti

        # 3. Mock Character Response Chain
        mock_char_chain = MagicMock()
        mock_char_chain.ainvoke = AsyncMock(return_value="Here is a picture for you!")
        mock_char_fn.return_value = mock_char_chain

        # 4. Mock Memory Manager
        mock_mm = MagicMock()
        mock_mm.get_relevant_memories.return_value = []
        mock_mm.format_memories_for_prompt.return_value = ""
        mock_mm.extract_and_store_memories = AsyncMock(return_value=None)
        mock_mm_fn.return_value = mock_mm

        # Execute Graph
        input_state = {"messages": [HumanMessage(content="Send me a photo!")]}
        result = await graph.ainvoke(input_state)

        # Assertions
        assert result.get("workflow") == "image"
        assert result.get("image_path") is not None
        assert "image_" in result.get("image_path")
        last_message = result["messages"][-1]
        assert isinstance(last_message, AIMessage)
        assert last_message.content == "Here is a picture for you!"


@pytest.mark.asyncio
async def test_graph_audio_execution_mocked():
    """Test full graph execution for audio workflow."""
    with (
        patch("src.graph.nodes.get_router_chain") as mock_router_fn,
        patch("src.graph.nodes.get_character_response_chain") as mock_char_fn,
        patch("src.graph.nodes.get_text_to_speech_module") as mock_tts_fn,
        patch("src.graph.nodes.get_memory_manager") as mock_mm_fn,
    ):
        # 1. Mock Router to select audio
        mock_router_chain = MagicMock()
        mock_router_chain.ainvoke = AsyncMock(
            return_value=RouterResponse(response_type="audio")
        )
        mock_router_fn.return_value = mock_router_chain

        # 2. Mock TTS Module
        mock_tts = MagicMock()
        mock_tts.synthesize = AsyncMock(return_value=b"fake_audio_mp3_data")
        mock_tts_fn.return_value = mock_tts

        # 3. Mock Character Response Chain
        mock_char_chain = MagicMock()
        mock_char_chain.ainvoke = AsyncMock(return_value="Listening to my voice note!")
        mock_char_fn.return_value = mock_char_chain

        # 4. Mock Memory Manager
        mock_mm = MagicMock()
        mock_mm.get_relevant_memories.return_value = []
        mock_mm.format_memories_for_prompt.return_value = ""
        mock_mm.extract_and_store_memories = AsyncMock(return_value=None)
        mock_mm_fn.return_value = mock_mm

        # Execute Graph
        input_state = {"messages": [HumanMessage(content="Send me an audio note!")]}
        result = await graph.ainvoke(input_state)

        # Assertions
        assert result.get("workflow") == "audio"
        assert result.get("audio_buffer") == b"fake_audio_mp3_data"
        last_message = result["messages"][-1]
        assert isinstance(last_message, AIMessage)
        assert last_message.content == "Listening to my voice note!"


@pytest.mark.asyncio
async def test_graph_summarization_trigger():
    """Test that graph automatically triggers summarization when messages exceed threshold."""
    trigger = settings.TOTAL_MESSAGES_SUMMARY_TRIGGER
    # Create messages list exceeding trigger
    messages = [HumanMessage(content=f"Msg {i}", id=f"id-{i}") for i in range(trigger + 2)]

    with (
        patch("src.graph.nodes.get_router_chain") as mock_router_fn,
        patch("src.graph.nodes.get_character_response_chain") as mock_char_fn,
        patch("src.graph.nodes.get_chat_model") as mock_chat_fn,
        patch("src.graph.nodes.get_memory_manager") as mock_mm_fn,
    ):
        mock_router_chain = MagicMock()
        mock_router_chain.ainvoke = AsyncMock(
            return_value=RouterResponse(response_type="conversation")
        )
        mock_router_fn.return_value = mock_router_chain

        mock_char_chain = MagicMock()
        mock_char_chain.ainvoke = AsyncMock(return_value="Latest reply")
        mock_char_fn.return_value = mock_char_chain

        mock_summarizer_model = MagicMock()
        mock_summarizer_model.ainvoke = AsyncMock(
            return_value=AIMessage(content="Summary of earlier conversation")
        )
        mock_chat_fn.return_value = mock_summarizer_model

        mock_mm = MagicMock()
        mock_mm.get_relevant_memories.return_value = []
        mock_mm.format_memories_for_prompt.return_value = ""
        mock_mm.extract_and_store_memories = AsyncMock(return_value=None)
        mock_mm_fn.return_value = mock_mm

        result = await graph.ainvoke({"messages": messages})

        assert result.get("summary") == "Summary of earlier conversation"
