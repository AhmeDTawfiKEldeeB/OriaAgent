from langchain_core.messages import HumanMessage
from langgraph.graph import END

from src.config.settings import settings
from src.graph.edges import select_workflow, should_summarize_conversation
from src.graph.state import OriaState


def test_should_summarize_conversation_below_threshold():
    state: OriaState = {
        "messages": [HumanMessage(content="Hi")],
        "summary": "",
        "workflow": "conversation",
        "audio_buffer": b"",
        "image_path": "",
        "current_activity": "",
        "apply_activity": False,
        "memory_context": "",
    }
    result = should_summarize_conversation(state)
    assert result == END


def test_should_summarize_conversation_above_threshold():
    trigger = settings.TOTAL_MESSAGES_SUMMARY_TRIGGER
    state: OriaState = {
        "messages": [HumanMessage(content=f"Message {i}") for i in range(trigger + 1)],
        "summary": "",
        "workflow": "conversation",
        "audio_buffer": b"",
        "image_path": "",
        "current_activity": "",
        "apply_activity": False,
        "memory_context": "",
    }
    result = should_summarize_conversation(state)
    assert result == "summarize_conversation_node"


def test_select_workflow_image():
    state: OriaState = {
        "messages": [],
        "summary": "",
        "workflow": "image",
        "audio_buffer": b"",
        "image_path": "",
        "current_activity": "",
        "apply_activity": False,
        "memory_context": "",
    }
    assert select_workflow(state) == "image_node"


def test_select_workflow_audio():
    state: OriaState = {
        "messages": [],
        "summary": "",
        "workflow": "audio",
        "audio_buffer": b"",
        "image_path": "",
        "current_activity": "",
        "apply_activity": False,
        "memory_context": "",
    }
    assert select_workflow(state) == "audio_node"


def test_select_workflow_conversation():
    state: OriaState = {
        "messages": [],
        "summary": "",
        "workflow": "conversation",
        "audio_buffer": b"",
        "image_path": "",
        "current_activity": "",
        "apply_activity": False,
        "memory_context": "",
    }
    assert select_workflow(state) == "conversation_node"
