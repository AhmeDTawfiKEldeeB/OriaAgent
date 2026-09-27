from src.graph.graph import create_oria_graph, graph


def test_oria_graph_structure():
    builder = create_oria_graph()
    assert builder is not None

    compiled = builder.compile()
    nodes = list(compiled.nodes.keys())

    expected_nodes = [
        "__start__",
        "context_injection_node",
        "memory_injection_node",
        "router_node",
        "conversation_node",
        "image_node",
        "audio_node",
        "memory_extraction_node",
        "summarize_conversation_node",
    ]
    for node in expected_nodes:
        assert node in nodes


def test_exported_graph_instance():
    assert graph is not None
    assert "__start__" in graph.nodes
