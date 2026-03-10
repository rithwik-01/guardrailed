import logging
import re
import string
from typing import Any, Dict, List, Tuple

from src.exceptions import ValidationError

logger = logging.getLogger(__name__)


def get_messages(request_json: Dict[str, Any]) -> List[Dict[str, str]]:
    if "messages" not in request_json:
        raise ValidationError("You must provide 'messages' field.")
    messages = request_json["messages"]
    if not isinstance(messages, list):
        raise ValidationError("The 'messages' field must be a list.")

    validated_messages = []
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValidationError(f"Message at index {idx} must be a dictionary.")
        role = msg.get("role")
        if role is None:
            raise ValidationError(f"Message at index {idx} is missing the 'role' key.")
        if "content" not in msg:
            raise ValidationError(
                f"Message at index {idx} is missing the 'content' key."
            )
        if not isinstance(role, str):
            raise ValidationError(f"The 'role' field at index {idx} must be a string.")
        if not isinstance(msg["content"], str):
            raise ValidationError(
                f"The 'content' field at index {idx} must be a string."
            )

        user_id = msg.get("user_id")
        if user_id is not None and not isinstance(user_id, str):
            raise ValidationError(
                f"The 'user_id' field at index {idx} must be a string."
            )

        validated_message = {"role": role, "content": msg["content"]}
        if user_id is not None:
            validated_message["user_id"] = user_id
        validated_messages.append(validated_message)

    return validated_messages


def chunk_text_by_char(
    text: str, max_chars: int, overlap_chars: int
) -> List[Tuple[str, int]]:
    """
    Chunks text by character count with overlap.

    Args:
        text: The input text string.
        max_chars: Maximum characters per chunk. Must be positive.
        overlap_chars: Number of characters to overlap between chunks. Must be non-negative and less than max_chars.

    Returns:
        A list of tuples, where each tuple is (chunk_text, original_start_index).
        Returns [(text, 0)] if chunking is not needed or inputs are invalid.
    """
    if not isinstance(text, str) or not text:
        return []
    if not isinstance(max_chars, int) or max_chars <= 0:
        logger.error(f"Invalid max_chars ({max_chars}). Must be a positive integer.")
        return [(text, 0)]
    if (
        not isinstance(overlap_chars, int)
        or overlap_chars < 0
        or overlap_chars >= max_chars
    ):
        logger.error(
            f"Invalid overlap_chars ({overlap_chars}). Must be >= 0 and < max_chars ({max_chars})."
        )
        return [(text, 0)]

    text_len = len(text)
    if text_len <= max_chars:
        return [(text, 0)]

    chunks_dict = {}
    start_index = 0
    stride = max(1, max_chars - overlap_chars)

    while start_index < text_len:
        end_index = min(start_index + max_chars, text_len)
        chunk_text = text[start_index:end_index]
        if chunk_text:
            chunks_dict[start_index] = chunk_text

        if end_index == text_len:
            break

        start_index += stride

    last_processed_end = (
        max(k + len(v) for k, v in chunks_dict.items()) if chunks_dict else 0
    )

    if last_processed_end < text_len:
        final_start = max(0, text_len - max_chars)
        if final_start not in chunks_dict:
            final_chunk = text[final_start:text_len]
            if final_chunk:
                chunks_dict[final_start] = final_chunk

    sorted_chunks = sorted(chunks_dict.items())
    result_list = [(text, start) for start, text in sorted_chunks]

    return result_list


def normalize_text(text: str) -> str:
    """Basic normalization: lowercase, strip whitespace, remove punctuation."""
    text = text.lower()
    text = text.strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text
