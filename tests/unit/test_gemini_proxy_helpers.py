from unittest.mock import MagicMock

from src.presentation.routes.gemini_proxy import (
    _extract_gemini_api_key,
    _extract_input_messages_from_gemini,
    _extract_output_message_from_gemini,
)
from src.shared import Agent


def test_extract_gemini_api_key_from_header():
    mock_request = MagicMock()
    mock_request.headers.get.return_value = "header_api_key"
    mock_request.query_params.get.return_value = None
    assert _extract_gemini_api_key(mock_request) == "header_api_key"
    mock_request.headers.get.assert_called_once_with("x-goog-api-key")
    mock_request.query_params.get.assert_not_called()


def test_extract_gemini_api_key_from_query():
    mock_request = MagicMock()
    mock_request.headers.get.return_value = None
    mock_request.query_params.get.return_value = "query_api_key"
    assert _extract_gemini_api_key(mock_request) == "query_api_key"
    mock_request.headers.get.assert_called_once_with("x-goog-api-key")
    mock_request.query_params.get.assert_called_once_with("key")


def test_extract_gemini_api_key_prefer_header():
    mock_request = MagicMock()
    mock_request.headers.get.return_value = "header_api_key"
    mock_request.query_params.get.return_value = "query_api_key"
    assert _extract_gemini_api_key(mock_request) == "header_api_key"
    mock_request.headers.get.assert_called_once_with("x-goog-api-key")
    mock_request.query_params.get.assert_not_called()


def test_extract_gemini_api_key_none():
    mock_request = MagicMock()
    mock_request.headers.get.return_value = None
    mock_request.query_params.get.return_value = None
    assert _extract_gemini_api_key(mock_request) is None


def test_extract_input_messages_valid():
    request_data = {
        "contents": [
            {"role": "user", "parts": [{"text": "Hello there."}]},
            {
                "role": "model",
                "parts": [{"text": "General Kenobi!"}],
            },  # Model input part
            {
                "role": "user",
                "parts": [{"text": "First part.\nSecond part."}],
            },  # Multi-part text
        ]
    }
    expected_messages = [
        {"role": Agent.USER, "content": "Hello there."},
        {"role": Agent.ASSISTANT, "content": "General Kenobi!"},
        {"role": Agent.USER, "content": "First part.\nSecond part."},
    ]
    assert _extract_input_messages_from_gemini(request_data) == expected_messages


def test_extract_input_messages_missing_contents():
    request_data = {"generationConfig": {}}
    assert _extract_input_messages_from_gemini(request_data) == []


def test_extract_input_messages_contents_not_list():
    request_data = {"contents": {"role": "user", "parts": []}}
    assert _extract_input_messages_from_gemini(request_data) == []


def test_extract_input_messages_invalid_item_in_contents():
    request_data = {"contents": ["not a dict"]}
    assert _extract_input_messages_from_gemini(request_data) == []


def test_extract_input_messages_missing_parts():
    request_data = {"contents": [{"role": "user"}]}
    assert _extract_input_messages_from_gemini(request_data) == []


def test_extract_input_messages_parts_not_list():
    request_data = {"contents": [{"role": "user", "parts": {"text": "hi"}}]}
    assert _extract_input_messages_from_gemini(request_data) == []


def test_extract_input_messages_part_not_dict():
    request_data = {"contents": [{"role": "user", "parts": ["just text"]}]}
    assert _extract_input_messages_from_gemini(request_data) == []


def test_extract_input_messages_part_missing_text():
    request_data = {"contents": [{"role": "user", "parts": [{"image_data": "..."}]}]}
    assert _extract_input_messages_from_gemini(request_data) == []


def test_extract_input_messages_empty_text():
    request_data = {
        "contents": [{"role": "user", "parts": [{"text": ""}, {"text": "  "}]}]
    }
    assert _extract_input_messages_from_gemini(request_data) == []


def test_extract_output_message_valid():
    response_data = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": "This is the response."}],
                },
                "finishReason": "STOP",
                "index": 0,
                "safetyRatings": [],
            }
        ],
        "promptFeedback": {},
    }
    expected_message = {"role": Agent.ASSISTANT, "content": "This is the response."}
    assert _extract_output_message_from_gemini(response_data) == expected_message


def test_extract_output_message_multiple_parts():
    response_data = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": "Part 1. "}, {"text": "Part 2."}],
                },
                "finishReason": "STOP",
            }
        ]
    }
    expected_message = {"role": Agent.ASSISTANT, "content": "Part 1. \nPart 2."}
    assert _extract_output_message_from_gemini(response_data) == expected_message


def test_extract_output_message_google_safety_block():
    response_data = {
        "candidates": [
            {
                "finishReason": "SAFETY",
                "index": 0,
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "probability": "HIGH",
                    }
                ],
            }
        ],
        "promptFeedback": {"blockReason": "SAFETY"},
    }
    assert _extract_output_message_from_gemini(response_data) is None


def test_extract_output_message_no_candidates():
    response_data = {"promptFeedback": {}}
    assert _extract_output_message_from_gemini(response_data) is None


def test_extract_output_message_candidates_not_list():
    response_data = {"candidates": {"content": {}}}
    assert _extract_output_message_from_gemini(response_data) is None


def test_extract_output_message_no_content():
    response_data = {"candidates": [{"finishReason": "STOP"}]}
    assert _extract_output_message_from_gemini(response_data) is None


def test_extract_output_message_content_not_dict():
    response_data = {"candidates": [{"content": ["parts"], "finishReason": "STOP"}]}
    assert _extract_output_message_from_gemini(response_data) is None


def test_extract_output_message_no_parts():
    response_data = {
        "candidates": [{"content": {"role": "model"}, "finishReason": "STOP"}]
    }
    assert _extract_output_message_from_gemini(response_data) is None


def test_extract_output_message_parts_not_list():
    response_data = {
        "candidates": [
            {
                "content": {"role": "model", "parts": {"text": "hi"}},
                "finishReason": "STOP",
            }
        ]
    }
    assert _extract_output_message_from_gemini(response_data) is None


def test_extract_output_message_part_not_dict():
    response_data = {
        "candidates": [
            {"content": {"role": "model", "parts": ["hi"]}, "finishReason": "STOP"}
        ]
    }
    assert _extract_output_message_from_gemini(response_data) is None


def test_extract_output_message_part_no_text():
    response_data = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"image": "..."}]},
                "finishReason": "STOP",
            }
        ]
    }
    assert _extract_output_message_from_gemini(response_data) is None
