import string

import pytest

from src.exceptions import ValidationError
from src.utils import chunk_text_by_char, get_messages, normalize_text


class TestGetMessages:
    """Tests for the get_messages function."""

    def test_valid_messages(self):
        """Test that valid messages are properly validated and returned."""
        request = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
        }
        expected = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        assert get_messages(request) == expected

    def test_messages_with_user_id(self):
        """Test that user_id field is preserved when present."""
        request = {
            "messages": [
                {"role": "user", "content": "Hello", "user_id": "user123"},
                {"role": "assistant", "content": "Hi there"},
            ]
        }
        expected = [
            {"role": "user", "content": "Hello", "user_id": "user123"},
            {"role": "assistant", "content": "Hi there"},
        ]
        assert get_messages(request) == expected

    def test_messages_with_extra_fields(self):
        """Test that extra fields are removed during validation."""
        request = {
            "messages": [
                {"role": "user", "content": "Hello", "extra_field": "value"},
                {"role": "assistant", "content": "Hi there", "ignored": True},
            ]
        }
        expected = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        assert get_messages(request) == expected

    def test_messages_with_mixed_fields(self):
        """Test validation with mixed fields across messages."""
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                    "user_id": "user123",
                    "extra": "ignored",
                },
                {"role": "assistant", "content": "Hi there"},
                {
                    "role": "user",
                    "content": "How are you?",
                    "metadata": {"ignored": True},
                },
            ]
        }
        expected = [
            {"role": "user", "content": "Hello", "user_id": "user123"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]
        assert get_messages(request) == expected

    def test_empty_messages_list(self):
        """Test that an empty messages list is valid."""
        request = {"messages": []}
        assert get_messages(request) == []

    def test_missing_messages_field(self):
        """Test that ValidationError is raised when 'messages' field is missing."""
        request = {"other_field": "value"}
        with pytest.raises(ValidationError, match="You must provide 'messages' field"):
            get_messages(request)

    def test_messages_not_list(self):
        """Test that ValidationError is raised when 'messages' is not a list."""
        request = {"messages": "not a list"}
        with pytest.raises(
            ValidationError, match="The 'messages' field must be a list"
        ):
            get_messages(request)

    def test_message_not_dict(self):
        """Test that ValidationError is raised when a message is not a dictionary."""
        request = {"messages": ["not a dict"]}
        with pytest.raises(
            ValidationError, match="Message at index 0 must be a dictionary"
        ):
            get_messages(request)

    def test_missing_role(self):
        """Test that ValidationError is raised when 'role' is missing."""
        request = {"messages": [{"content": "Hello"}]}
        with pytest.raises(
            ValidationError, match="Message at index 0 is missing the 'role' key"
        ):
            get_messages(request)

    def test_missing_content(self):
        """Test that ValidationError is raised when 'content' is missing."""
        request = {"messages": [{"role": "user"}]}
        with pytest.raises(
            ValidationError, match="Message at index 0 is missing the 'content' key"
        ):
            get_messages(request)

    def test_role_not_string(self):
        """Test that ValidationError is raised when 'role' is not a string."""
        request = {"messages": [{"role": 123, "content": "Hello"}]}
        with pytest.raises(
            ValidationError, match="The 'role' field at index 0 must be a string"
        ):
            get_messages(request)

    def test_content_not_string(self):
        """Test that ValidationError is raised when 'content' is not a string."""
        request = {"messages": [{"role": "user", "content": 123}]}
        with pytest.raises(
            ValidationError, match="The 'content' field at index 0 must be a string"
        ):
            get_messages(request)

    def test_user_id_not_string(self):
        """Test that ValidationError is raised when 'user_id' is not a string."""
        request = {"messages": [{"role": "user", "content": "Hello", "user_id": 123}]}
        with pytest.raises(
            ValidationError, match="The 'user_id' field at index 0 must be a string"
        ):
            get_messages(request)


class TestChunkTextByChar:
    """Tests for the chunk_text_by_char function."""

    def test_empty_text(self):
        """Test chunking with empty text."""
        result = chunk_text_by_char("", 10, 2)
        assert result == []

    def test_none_text(self):
        """Test chunking with None text."""
        result = chunk_text_by_char(None, 10, 2)
        assert result == []

    def test_text_shorter_than_max_chars(self):
        """Test chunking when text is shorter than max_chars."""
        text = "Short text"
        result = chunk_text_by_char(text, 20, 5)
        assert result == [(text, 0)]

    def test_text_equal_to_max_chars(self):
        """Test chunking when text length equals max_chars."""
        text = "Exactly 10"  # 10 characters
        result = chunk_text_by_char(text, 10, 2)
        assert result == [(text, 0)]

    def test_basic_chunking_no_overlap(self):
        """Test basic chunking with no overlap."""
        text = "This is a test for chunking by character count."
        max_chars = 10
        overlap = 0

        expected = [
            ("This is a ", 0),
            ("test for c", 10),
            ("hunking by", 20),
            (" character", 30),
            (" count.", 40),
        ]

        result = chunk_text_by_char(text, max_chars, overlap)
        assert result == expected

    def test_chunking_with_overlap(self):
        """Test chunking with overlap."""
        text = "This is a test for chunking with overlap."
        max_chars = 10
        overlap = 3

        result = chunk_text_by_char(text, max_chars, overlap)

        assert result[0][1] == 0

        for i, (chunk, _) in enumerate(result[:-1]):
            assert len(chunk) <= max_chars, f"Chunk {i} exceeds max_chars: {chunk}"

        for i in range(len(result) - 1):
            curr_start = result[i][1]
            next_start = result[i + 1][1]
            expected_stride = max_chars - overlap
            assert (
                abs(next_start - curr_start - expected_stride) <= 1
            ), f"Stride between chunks {i} and {i+1} should be close to {expected_stride}"

        assert (
            text[-5:] in result[-1][0]
        ), "Last chunk should include the end of the text"

    def test_very_small_stride(self):
        """Test chunking with very small stride (overlap almost equal to max_chars)."""
        text = "This is a test for very small stride."
        max_chars = 10
        overlap = 9

        result = chunk_text_by_char(text, max_chars, overlap)

        for i, (chunk, _) in enumerate(result[:-1]):
            assert len(chunk) <= max_chars, f"Chunk {i} exceeds max_chars: {chunk}"

        for i in range(len(result) - 1):
            curr_start = result[i][1]
            next_start = result[i + 1][1]
            assert (
                next_start - curr_start == 1
            ), f"Stride between chunks {i} and {i+1} is not 1"

    def test_chunking_last_chunk_alignment(self):
        """Test that the last chunk is properly included."""
        text = "This is a test to ensure the last chunk is included properly!"
        max_chars = 15
        overlap = 5

        result = chunk_text_by_char(text, max_chars, overlap)

        last_char_included = any(chunk[0][-1] == text[-1] for chunk in result)
        assert (
            last_char_included
        ), "The last character should be included in at least one chunk"

        assert result[-1][0].endswith(
            text[-5:]
        ), "The last chunk should include the end of the text"

    def test_chunk_deduplication(self):
        """Test that duplicate chunks (by start index) are handled correctly."""
        text = "Short text for testing deduplication"
        max_chars = 10
        overlap = 9

        result = chunk_text_by_char(text, max_chars, overlap)

        start_indices = [chunk[1] for chunk in result]
        assert len(start_indices) == len(
            set(start_indices)
        ), "There should be no duplicate start indices"

    def test_large_text(self):
        """Test chunking with a very large text."""
        text = "a" * 10000
        max_chars = 1000
        overlap = 100

        result = chunk_text_by_char(text, max_chars, overlap)

        expected_chunks = 11
        assert (
            len(result) >= expected_chunks
        ), f"Expected at least {expected_chunks} chunks, got {len(result)}"

        first_chunk_start = result[0][1]
        last_chunk_end = result[-1][1] + len(result[-1][0])

        assert first_chunk_start == 0, "First chunk should start at position 0"
        assert last_chunk_end >= len(
            text
        ), "Last chunk should cover the end of the text"

        first_chunk_start = result[0][1]
        last_chunk_end = result[-1][1] + len(result[-1][0])

        assert first_chunk_start == 0, "First chunk should start at position 0"
        assert last_chunk_end == len(
            text
        ), "Last chunk should end at the end of the text"

    def test_invalid_max_chars(self):
        """Test chunking with invalid max_chars."""
        result = chunk_text_by_char("Test text", -5, 2)
        assert result == [("Test text", 0)]

        result = chunk_text_by_char("Test text", 0, 2)
        assert result == [("Test text", 0)]

        result = chunk_text_by_char("Test text", "10", 2)
        assert result == [("Test text", 0)]

    def test_invalid_overlap_chars(self):
        """Test chunking with invalid overlap_chars."""
        result = chunk_text_by_char("Test text", 10, -2)
        assert result == [("Test text", 0)]

        result = chunk_text_by_char("Test text", 10, 10)
        assert result == [("Test text", 0)]

        result = chunk_text_by_char("Test text", 10, 15)
        assert result == [("Test text", 0)]

        result = chunk_text_by_char("Test text", 10, "2")
        assert result == [("Test text", 0)]


class TestNormalizeText:
    """Tests for the normalize_text function."""

    def test_basic_normalization(self):
        """Test basic text normalization."""
        text = "  This is a Test String.  "
        expected = "this is a test string"
        assert normalize_text(text) == expected

    def test_lowercase_conversion(self):
        """Test that text is properly converted to lowercase."""
        text = "ALL UPPERCASE TEXT"
        expected = "all uppercase text"
        assert normalize_text(text) == expected

    def test_punctuation_removal(self):
        """Test that punctuation is properly removed."""
        text = "Hello, world! This has punctuation: comma, period, etc."
        expected = "hello world this has punctuation comma period etc"
        assert normalize_text(text) == expected

        text = f"Test with {string.punctuation} punctuation"
        expected = "test with punctuation"
        assert normalize_text(text) == expected

    def test_whitespace_normalization(self):
        """Test that whitespace is properly normalized."""
        text = "Multiple    spaces    between    words"
        expected = "multiple spaces between words"
        assert normalize_text(text) == expected

        text = "Text with\ttabs\tand\nnewlines\n\n"
        expected = "text with tabs and newlines"
        assert normalize_text(text) == expected

        text = "  Mixed   \t whitespace \n  example  "
        expected = "mixed whitespace example"
        assert normalize_text(text) == expected

    def test_empty_string(self):
        """Test normalization of an empty string."""
        text = ""
        expected = ""
        assert normalize_text(text) == expected

    def test_only_whitespace(self):
        """Test normalization of string with only whitespace."""
        text = "   \t\n  "
        expected = ""
        assert normalize_text(text) == expected

    def test_only_punctuation(self):
        """Test normalization of string with only punctuation."""
        text = ".,!?;:"
        expected = ""
        assert normalize_text(text) == expected

    def test_unicode_characters(self):
        """Test normalization with Unicode characters."""
        text = "Café Müller în București"
        expected = "café müller în bucurești"
        assert normalize_text(text) == expected

    def test_special_characters(self):
        """Test normalization with special characters."""
        text = "Test with special chars: © ® ™ £ € ¥"
        normalized = normalize_text(text)

        assert normalized.startswith("test with special chars")
        assert normalized == normalized.lower()
        assert ":" not in normalized
        for char in ["©", "®", "™", "£", "€", "¥"]:
            assert char in normalized
