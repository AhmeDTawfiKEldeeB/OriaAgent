from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from src.core.prompts import CHARACTER_CARD_PROMPT, ROUTER_PROMPT
from src.graph.utils.helpers import AsteriskRemovalParser, get_chat_model


class RouterResponse(BaseModel):
    response_type: str = Field(
        description="The response type to give to the user. It must be one of: 'conversation', 'image' or 'audio'"
    )


def get_router_chain() -> Runnable:
    """Create and return the router chain that classifies user intent into conversation, image, or audio."""
    model = get_chat_model(temperature=0.3).with_structured_output(RouterResponse)

    prompt = ChatPromptTemplate.from_messages(
        [("system", ROUTER_PROMPT), MessagesPlaceholder(variable_name="messages")]
    )

    return prompt | model


def get_character_response_chain(
    summary: str = "",
    memory_context: str = "",
    current_activity: str = "",
) -> Runnable:
    """Create and return the character response chain for Oria.

    Injects earlier summary, memories, and current activity into the character card.
    """
    model = get_chat_model()
    system_message = CHARACTER_CARD_PROMPT

    if summary:
        system_message += f"\n\nSummary of conversation earlier between Oria and the user: {summary}"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages"),
        ]
    ).partial(
        memory_context=memory_context,
        current_activity=current_activity,
    )

    return prompt | model | AsteriskRemovalParser()
