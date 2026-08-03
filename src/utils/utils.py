import logging
import re
import string
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from src.exceptions import ValidationError

logger = logging.getLogger(__name__)

# Unicode blocks used to smuggle text past text classifiers. Stripping these is
# lossless for detection purposes: none of them carry visible meaning.
# See "Bypassing LLM Guardrails" (arXiv:2504.11168) for the attack classes.
_TAG_BLOCK = range(0xE0000, 0xE0080)  # invisible Unicode tag chars
_VARIATION_SELECTORS = list(range(0xFE00, 0xFE10)) + list(range(0xE0100, 0xE01F0))
_INVISIBLE_CODEPOINTS = frozenset(
    list(_TAG_BLOCK)
    + _VARIATION_SELECTORS
    + [
        0x00AD,  # soft hyphen
        0x180E,  # mongolian vowel separator
        0x200B,  # zero width space
        0x200C,  # zero width non-joiner
        0x200D,  # zero width joiner
        0x2060,  # word joiner
        0xFEFF,  # zero width no-break space / BOM
    ]
)

# Latin lookalikes from other scripts (Cyrillic, Greek, fullwidth, math alphanumerics
# are handled by NFKC). Folding these is lossy for genuine non-Latin text, so it is
# only ever applied to the copy handed to detectors, never to forwarded content.
_CONFUSABLES = {
    "а": "a",
    "е": "e",
    "о": "o",
    "р": "p",
    "с": "c",
    "у": "y",
    "х": "x",
    "і": "i",
    "ј": "j",
    "һ": "h",
    "А": "A",
    "В": "B",
    "Е": "E",
    "К": "K",
    "М": "M",
    "Н": "H",
    "О": "O",
    "Р": "P",
    "С": "C",
    "Т": "T",
    "Х": "X",
    "Ѕ": "S",
    "І": "I",
    "Ј": "J",
    "α": "a",
    "ο": "o",
    "ρ": "p",
    "υ": "u",
    "ν": "v",
    "Α": "A",
    "Β": "B",
    "Ε": "E",
    "Ζ": "Z",
    "Η": "H",
    "Ι": "I",
    "Κ": "K",
    "Μ": "M",
    "Ν": "N",
    "Ο": "O",
    "Ρ": "P",
    "Τ": "T",
    "Υ": "Y",
    "Χ": "X",
    "ı": "i",
    "ɡ": "g",
    "ǀ": "l",
    "⁄": "/",
    "∕": "/",
}
_CONFUSABLES_TABLE = str.maketrans(_CONFUSABLES)


def strip_invisible(text: str) -> str:
    """
    Remove invisible characters and apply NFKC normalization.

    Strips zero-width characters, bidirectional overrides, the Unicode tag block
    and variation selectors, then applies NFKC so that fullwidth, superscript and
    mathematical alphanumeric variants collapse to their canonical form.

    This transformation removes no legible content, so the result is safe both to
    scan and to forward upstream.
    """
    if not text:
        return text
    cleaned = "".join(
        ch
        for ch in text
        if ord(ch) not in _INVISIBLE_CODEPOINTS
        and unicodedata.category(ch) not in ("Cf", "Co", "Cs")
    )
    return unicodedata.normalize("NFKC", cleaned)


def sanitize_for_detection(text: str, fold_homoglyphs: bool = True) -> str:
    """
    Produce the text that policy checks should run against.

    Applies :func:`strip_invisible`, then optionally folds cross-script homoglyphs
    (Cyrillic/Greek lookalikes) down to their Latin equivalents.

    The homoglyph fold is lossy for genuine Cyrillic or Greek text, so the result
    is for detection only and must never be forwarded to an upstream provider or
    returned to the caller.
    """
    cleaned = strip_invisible(text)
    if fold_homoglyphs and cleaned:
        cleaned = cleaned.translate(_CONFUSABLES_TABLE)
    return cleaned


def extract_text_content(content: Any) -> Optional[str]:
    """
    Flatten a chat message 'content' field into a single text string.

    Handles the plain-string form and the content-parts list form used by the
    OpenAI and Anthropic APIs (``[{"type": "text", "text": "..."}]``). Parts that
    carry no text (images, audio) contribute nothing and are logged.

    Returns:
        The extracted text, or None if the content is of a shape we cannot read
        (callers must treat None as unvalidatable and fail closed).
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts: List[str] = []
        for part in content:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    texts.append(text)
                else:
                    logger.warning(
                        f"Content part of type '{part.get('type', 'unknown')}' carries "
                        "no text and cannot be validated."
                    )
            else:
                logger.warning(
                    f"Unsupported content part type {type(part)}; skipping part."
                )
        return "\n".join(texts)

    return None


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
