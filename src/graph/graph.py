from langgraph.graph import END, START, StateGraph

from src.graph.edges import select_workflow, should_summarize_conversation
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


def create_oria_graph() -> StateGraph:
    """Construct and assemble the Oria agent state graph."""
    builder = StateGraph(OriaState)

    # 1. Add Nodes
    builder.add_node("context_injection_node", context_injection_node)
    builder.add_node("memory_injection_node", memory_injection_node)
    builder.add_node("router_node", router_node)
    builder.add_node("conversation_node", conversation_node)
    builder.add_node("image_node", image_node)
    builder.add_node("audio_node", audio_node)
    builder.add_node("memory_extraction_node", memory_extraction_node)
    builder.add_node("summarize_conversation_node", summarize_conversation_node)

    # 2. Add Fixed Edges
    builder.add_edge(START, "context_injection_node")
    builder.add_edge("context_injection_node", "memory_injection_node")
    builder.add_edge("memory_injection_node", "router_node")

    # 3. Add Conditional Edge for Workflow Branching
    builder.add_conditional_edges(
        "router_node",
        select_workflow,
        {
            "conversation_node": "conversation_node",
            "image_node": "image_node",
            "audio_node": "audio_node",
        },
    )

    # 4. Route Responses to Memory Extraction
    builder.add_edge("conversation_node", "memory_extraction_node")
    builder.add_edge("image_node", "memory_extraction_node")
    builder.add_edge("audio_node", "memory_extraction_node")

    # 5. Add Conditional Edge for Conversation Summarization
    builder.add_conditional_edges(
        "memory_extraction_node",
        should_summarize_conversation,
        {
            "summarize_conversation_node": "summarize_conversation_node",
            END: END,
        },
    )

    # 6. Summarizer routes to END
    builder.add_edge("summarize_conversation_node", END)

    return builder


# Compile the graph instance for LangGraph Studio and production runtime
graph = create_oria_graph().compile()

__all__ = ["create_oria_graph", "graph"]
