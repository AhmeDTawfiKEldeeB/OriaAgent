from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage
from src.graph.utils.helpers import (
    AsteriskRemovalParser,
    remove_asterisk_content,
    get_text_to_image_module,
    get_image_to_text_module,
)
from src.graph.utils.chains import RouterResponse, get_router_chain, get_character_response_chain


def test_remove_asterisk_content():
    text = "Hello there! *smiles softly* How can I help you? *winks*"
    cleaned = remove_asterisk_content(text)
    assert cleaned == "Hello there!  How can I help you?"
    assert "*" not in cleaned


def test_asterisk_removal_parser():
    parser = AsteriskRemovalParser()
    parsed = parser.parse("*laughs* That was funny!")
    assert parsed == "That was funny!"


def test_router_response_model():
    res = RouterResponse(response_type="conversation")
    assert res.response_type == "conversation"

    res_img = RouterResponse(response_type="image")
    assert res_img.response_type == "image"


@patch("src.graph.utils.chains.get_chat_model")
def test_get_router_chain(mock_get_chat_model):
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value = mock_model
    mock_get_chat_model.return_value = mock_model

    chain = get_router_chain()
    assert chain is not None
    mock_get_chat_model.assert_called_once_with(temperature=0.3)
    mock_model.with_structured_output.assert_called_once_with(RouterResponse)


@patch("src.graph.utils.chains.get_chat_model")
def test_get_character_response_chain(mock_get_chat_model):
    mock_model = MagicMock()
    mock_get_chat_model.return_value = mock_model

    chain = get_character_response_chain(summary="Earlier talk")
    assert chain is not None
    mock_get_chat_model.assert_called_once()


def test_module_getters():
    img_module = get_text_to_image_module()
    assert img_module is not None

    itt_module = get_image_to_text_module()
    assert itt_module is not None
