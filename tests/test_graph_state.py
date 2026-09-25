from langchain_core.messages import HumanMessage
from src.graph.state import OriaState


def test_oria_state_annotations():
    annotations = OriaState.__annotations__
    assert "messages" in annotations
    assert annotations["summary"] is str
    assert annotations["workflow"] is str
    assert annotations["audio_buffer"] is bytes
    assert annotations["image_path"] is str
    assert annotations["current_activity"] is str
    assert annotations["apply_activity"] is bool
    assert annotations["memory_context"] is str


def test_oria_state_instantiation():
    state: OriaState = {
        "messages": [HumanMessage(content="Hello Oria")],
        "summary": "Greeting",
        "workflow": "conversation",
        "audio_buffer": b"",
        "image_path": "/tmp/img.png",
        "current_activity": "reading",
        "apply_activity": True,
        "memory_context": "Remembered user name",
    }
    assert state["workflow"] == "conversation"
    assert len(state["messages"]) == 1
