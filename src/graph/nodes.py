import os
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig

from src.config.settings import settings
from src.graph.state import OriaState
from src.graph.utils.chains import (
    get_character_response_chain,
    get_router_chain,
)
from src.graph.utils.helpers import (
    get_chat_model,
    get_text_to_image_module,
    get_text_to_speech_module,
)
from src.modules.memory import get_memory_manager
from src.modules.schedules import ScheduleContextGenerator


async def router_node(state: OriaState) -> dict:
    """Analyze recent conversation history and route to the appropriate workflow."""
    chain = get_router_chain()
    response = await chain.ainvoke(
        {"messages": state["messages"][-settings.ROUTER_MESSAGES_TO_ANALYZE :]}
    )
    return {"workflow": response.response_type}


def context_injection_node(state: OriaState) -> dict:
    """Check Oria's current schedule activity and flag if an activity update should apply."""
    schedule_context = ScheduleContextGenerator.get_current_activity()
    apply_activity = schedule_context != state.get("current_activity", "")
    return {"apply_activity": apply_activity, "current_activity": schedule_context}


async def conversation_node(state: OriaState, config: RunnableConfig) -> dict:
    """Generate a conversational response from Oria."""
    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "")

    chain = get_character_response_chain(state.get("summary", ""))

    response = await chain.ainvoke(
        {
            "messages": state["messages"],
            "current_activity": current_activity,
            "memory_context": memory_context,
        },
        config,
    )
    return {"messages": AIMessage(content=response)}


async def image_node(state: OriaState, config: RunnableConfig) -> dict:
    """Generate an image scenario and create an image corresponding to recent context."""
    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "")

    chain = get_character_response_chain(state.get("summary", ""))
    text_to_image_module = get_text_to_image_module()

    scenario = await text_to_image_module.create_scenario(state["messages"][-5:])
    os.makedirs("generated_images", exist_ok=True)
    img_path = f"generated_images/image_{str(uuid4())}.png"
    await text_to_image_module.generate_image(scenario.image_prompt, img_path)

    # Inject the image prompt information as context
    scenario_message = HumanMessage(
        content=f"<image attached by Oria generated from prompt: {scenario.image_prompt}>"
    )
    updated_messages = state["messages"] + [scenario_message]

    response = await chain.ainvoke(
        {
            "messages": updated_messages,
            "current_activity": current_activity,
            "memory_context": memory_context,
        },
        config,
    )

    return {"messages": AIMessage(content=response), "image_path": img_path}


async def audio_node(state: OriaState, config: RunnableConfig) -> dict:
    """Generate speech audio for Oria's response."""
    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "")

    chain = get_character_response_chain(state.get("summary", ""))
    text_to_speech_module = get_text_to_speech_module()

    response = await chain.ainvoke(
        {
            "messages": state["messages"],
            "current_activity": current_activity,
            "memory_context": memory_context,
        },
        config,
    )
    output_audio = await text_to_speech_module.synthesize(response)

    # Save audio to local generated_audio folder
    os.makedirs("generated_audio", exist_ok=True)
    audio_file_path = f"generated_audio/audio_{str(uuid4())}.mp3"
    with open(audio_file_path, "wb") as f:
        f.write(output_audio)

    return {"messages": AIMessage(content=response), "audio_buffer": output_audio}


async def summarize_conversation_node(state: OriaState) -> dict:
    """Summarize earlier conversation history and prune older messages."""
    model = get_chat_model()
    summary = state.get("summary", "")

    if summary:
        summary_message = (
            f"This is summary of the conversation to date between Oria and the user: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = (
            "Create a summary of the conversation above between Oria and the user. "
            "The summary must be a short description of the conversation so far, "
            "but that captures all the relevant information shared between Oria and the user:"
        )

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = await model.ainvoke(messages)

    delete_messages = [
        RemoveMessage(id=m.id)
        for m in state["messages"][: -settings.TOTAL_MESSAGES_AFTER_SUMMARY]
    ]
    return {"summary": response.content, "messages": delete_messages}


async def memory_extraction_node(state: OriaState) -> dict:
    """Extract and store important information from the latest user message."""
    messages = state.get("messages", [])
    if not messages:
        return {}

    # Extract memories from the most recent human message, ignoring AI responses
    last_human_message = next(
        (m for m in reversed(messages) if getattr(m, "type", "") == "human"),
        None,
    )
    if not last_human_message:
        return {}

    memory_manager = get_memory_manager()
    await memory_manager.extract_and_store_memories(last_human_message)
    return {}


def memory_injection_node(state: OriaState) -> dict:
    """Retrieve and inject relevant memories into the character card context."""
    memory_manager = get_memory_manager()

    # Get relevant memories based on recent conversation
    messages = state.get("messages", [])
    recent_context = " ".join([m.content for m in messages[-3:]]) if messages else ""
    memories = memory_manager.get_relevant_memories(recent_context)

    # Format memories for the character card
    memory_context = memory_manager.format_memories_for_prompt(memories)

    return {"memory_context": memory_context}


__all__ = [
    "router_node",
    "context_injection_node",
    "conversation_node",
    "image_node",
    "audio_node",
    "summarize_conversation_node",
    "memory_extraction_node",
    "memory_injection_node",
]