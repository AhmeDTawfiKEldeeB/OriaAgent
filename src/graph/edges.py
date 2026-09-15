from typing import Literal

from langgraph.graph import END

from src.config.settings import settings
from src.graph.state import OriaState


def should_summarize_conversation(
    state: OriaState,
) -> Literal["summarize_conversation_node", "__end__"]:
    """Determine whether the conversation exceeds the trigger limit and should be summarized."""
    messages = state["messages"]

    if len(messages) > settings.TOTAL_MESSAGES_SUMMARY_TRIGGER:
        return "summarize_conversation_node"

    return END


def select_workflow(
    state: OriaState,
) -> Literal["conversation_node", "image_node", "audio_node"]:
    """Select the appropriate workflow branch based on router output."""
    workflow = state.get("workflow", "conversation")

    if workflow == "image":
        return "image_node"

    elif workflow == "audio":
        return "audio_node"

    else:
        return "conversation_node"


__all__ = [
    "should_summarize_conversation",
    "select_workflow",
]
