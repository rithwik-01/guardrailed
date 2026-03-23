"""
Unit tests for the TTL Cache implementation.

This test suite verifies the correctness of the TTL cache implementation
including expiration, eviction, statistics tracking, and thread safety.

Test Categories:
1. Basic operations (get, put, clear)
2. TTL expiration behavior
3. LRU eviction when cache is full
4. Statistics tracking (hits, misses, evictions)
5. Cache key generation for validators
6. Integration with prompt injection validator
"""

import time
import threading
from unittest.mock import patch

import pytest

from src.utils.cache import (
    TTLCache,
    CacheEntry,
    generate_cache_key,
    get_injection_cache,
    reset_injection_cache,
)


class TestCacheEntry:
    """Tests for the CacheEntry class."""

    def test_cache_entry_creation(self):
        """Test that CacheEntry is created with correct timestamps."""
        entry = CacheEntry(value={"score": 0.95}, ttl_seconds=300)
        assert entry.value == {"score": 0.95}
        assert entry.expires_at > entry.created_at
        assert entry.expires_at - entry.created_at == 300
        assert not entry.is_expired()
        assert entry.age_seconds() >= 0
        assert entry.ttl_remaining() > 0

    def test_cache_entry_expiration(self):
        """Test that CacheEntry correctly detects expiration."""
        entry = CacheEntry(value="test", ttl_seconds=1)
        assert not entry.is_expired()
        time.sleep(1.1)
        assert entry.is_expired()

    def test_cache_entry_ttl_remaining(self):
        """Test TTL remaining calculation."""
        entry = CacheEntry(value="test", ttl_seconds=10)
        remaining = entry.ttl_remaining()
        assert 0 < remaining <= 10
        time.sleep(1)
        new_remaining = entry.ttl_remaining()
        assert new_remaining < remaining


class TestTTLCache:
    """Tests for the TTLCache class."""

    def test_cache_initialization(self):
        """Test cache initialization with valid parameters."""
        cache = TTLCache(maxsize=100, ttl=300, name="TestCache")
        assert cache.maxsize == 100
        assert cache.default_ttl == 300
        assert cache.name == "TestCache"
        assert cache.size() == 0
        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_cache_initialization_invalid_params(self):
        """Test cache initialization with invalid parameters."""
        with pytest.raises(ValueError, match="maxsize must be positive"):
            TTLCache(maxsize=0, ttl=300)

        with pytest.raises(ValueError, match="maxsize must be positive"):
            TTLCache(maxsize=-1, ttl=300)

        with pytest.raises(ValueError, match="ttl must be positive"):
            TTLCache(maxsize=100, ttl=0)

        with pytest.raises(ValueError, match="ttl must be positive"):
            TTLCache(maxsize=100, ttl=-1)

    def test_cache_put_and_get(self):
        """Test basic put and get operations."""
        cache = TTLCache(maxsize=10, ttl=60)

        # Put a value
        cache.put("key1", {"result": "INJECTION"})
        assert cache.size() == 1

        # Get the value back
        result = cache.get("key1")
        assert result == {"result": "INJECTION"}

        # Stats should show a hit
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0

    def test_cache_get_miss(self):
        """Test get operation with non-existent key."""
        cache = TTLCache(maxsize=10, ttl=60)

        result = cache.get("nonexistent")
        assert result is None

        stats = cache.stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0

    def test_cache_put_updates_existing(self):
        """Test that put updates existing key."""
        cache = TTLCache(maxsize=10, ttl=60)

        cache.put("key1", {"value": 1})
        cache.put("key1", {"value": 2})

        assert cache.size() == 1
        assert cache.get("key1") == {"value": 2}

    def test_cache_lru_eviction(self):
        """Test that least recently used entries are evicted when cache is full."""
        cache = TTLCache(maxsize=3, ttl=60)

        # Fill the cache
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        assert cache.size() == 3

        # Adding a 4th entry should evict the first (oldest)
        cache.put("key4", "value4")
        assert cache.size() == 3

        # key1 should have been evicted
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

        # Stats should show an eviction
        stats = cache.stats()
        assert stats["evictions"] == 1

    def test_cache_lru_order_updates_on_access(self):
        """Test that accessing an entry updates its LRU position."""
        cache = TTLCache(maxsize=3, ttl=60)

        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")

        # Access key1 to make it recently used
        cache.get("key1")

        # Add key4 - should evict key2 (now the oldest)
        cache.put("key4", "value4")

        assert cache.get("key1") == "value1"  # Still there
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_cache_ttl_expiration(self):
        """Test that entries expire after TTL."""
        cache = TTLCache(maxsize=10, ttl=1)  # 1 second TTL

        cache.put("key1", "value1")

        # Should be available immediately
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(1.5)

        # Should be expired now
        result = cache.get("key1")
        assert result is None

        # Stats should show a miss due to expiration
        stats = cache.stats()
        assert stats["expirations"] == 1

    def test_cache_custom_ttl(self):
        """Test put operation with custom TTL."""
        cache = TTLCache(maxsize=10, ttl=60)  # Default 60s

        # Put with custom TTL of 1 second
        cache.put("key1", "value1", ttl=1)

        # Should expire quickly
        time.sleep(1.5)
        assert cache.get("key1") is None

        # Default TTL should still work for other entries
        cache.put("key2", "value2")
        time.sleep(1.5)
        assert cache.get("key2") == "value2"  # Not expired

    def test_cache_clear(self):
        """Test clearing the cache."""
        cache = TTLCache(maxsize=10, ttl=60)

        cache.put("key1", "value1")
        cache.put("key2", "value2")
        assert cache.size() == 2

        cache.clear()
        assert cache.size() == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None

        # Stats should be reset
        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["evictions"] == 0

    def test_cache_cleanup_expired(self):
        """Test manual cleanup of expired entries."""
        cache = TTLCache(maxsize=10, ttl=1)

        # Add several entries
        for i in range(5):
            cache.put(f"key{i}", f"value{i}")

        assert cache.size() == 5

        # Wait for expiration
        time.sleep(1.5)

        # Manual cleanup
        removed = cache.cleanup_expired()
        assert removed == 5
        assert cache.size() == 0

    def test_cache_stats_hit_rate(self):
        """Test cache hit rate calculation."""
        cache = TTLCache(maxsize=10, ttl=60)

        cache.put("key1", "value1")

        # 3 hits
        cache.get("key1")
        cache.get("key1")
        cache.get("key1")

        # 1 miss
        cache.get("key2")

        stats = cache.stats()
        assert stats["hits"] == 3
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.75  # 3/4

    def test_cache_stats_empty(self):
        """Test stats when cache is empty."""
        cache = TTLCache(maxsize=10, ttl=60)
        stats = cache.stats()

        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["evictions"] == 0
        assert stats["expirations"] == 0
        assert stats["size"] == 0
        assert stats["maxsize"] == 10
        assert stats["hit_rate"] == 0.0

    def test_cache_thread_safety(self):
        """Test that cache is thread-safe."""
        cache = TTLCache(maxsize=100, ttl=60)
        errors = []

        def worker(worker_id):
            try:
                for i in range(100):
                    key = f"worker{worker_id}_key{i}"
                    cache.put(key, worker_id)
                    result = cache.get(key)
                    assert result == worker_id
            except Exception as e:
                errors.append((worker_id, e))

        # Create multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Check for errors
        assert len(errors) == 0, f"Thread safety errors: {errors}"

        # Cache should have some entries
        assert cache.size() > 0


class TestGenerateCacheKey:
    """Tests for cache key generation."""

    def test_cache_key_deterministic(self):
        """Test that identical inputs produce identical keys."""
        key1 = generate_cache_key("test content", 0.5, 7)
        key2 = generate_cache_key("test content", 0.5, 7)
        assert key1 == key2

    def test_cache_key_content_sensitive(self):
        """Test that different content produces different keys."""
        key1 = generate_cache_key("content A", 0.5, 7)
        key2 = generate_cache_key("content B", 0.5, 7)
        assert key1 != key2

    def test_cache_key_threshold_sensitive(self):
        """Test that different thresholds produce different keys."""
        key1 = generate_cache_key("test content", 0.5, 7)
        key2 = generate_cache_key("test content", 0.7, 7)
        assert key1 != key2

    def test_cache_key_policy_sensitive(self):
        """Test that different policy IDs produce different keys."""
        key1 = generate_cache_key("test content", 0.5, 7)
        key2 = generate_cache_key("test content", 0.5, 6)
        assert key1 != key2

    def test_cache_key_with_extra_params(self):
        """Test cache key generation with extra parameters."""
        key1 = generate_cache_key("test", 0.5, 7, extra={"model": "v1"})
        key2 = generate_cache_key("test", 0.5, 7, extra={"model": "v2"})
        assert key1 != key2

    def test_cache_key_hash_format(self):
        """Test that cache key is a hex string (SHA-256)."""
        key = generate_cache_key("test", 0.5, 7)
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 produces 32 bytes = 64 hex chars
        assert all(c in "0123456789abcdef" for c in key)


class TestGlobalInjectionCache:
    """Tests for the global injection cache singleton."""

    def test_get_injection_cache_singleton(self):
        """Test that get_injection_cache returns the same instance."""
        reset_injection_cache()  # Ensure clean state

        cache1 = get_injection_cache()
        cache2 = get_injection_cache()

        assert cache1 is cache2
        assert cache1.name == "PromptInjectionCache"
        assert cache1.maxsize == 1000
        assert cache1.default_ttl == 300

    def test_reset_injection_cache(self):
        """Test resetting the global cache."""
        cache = get_injection_cache()
        cache.put("test", "value")
        assert cache.size() == 1

        reset_injection_cache()

        cache_after_reset = get_injection_cache()
        # Should be a new instance or cleared
        assert cache_after_reset.size() == 0

    @patch("src.utils.cache._injection_cache", None)
    def test_get_injection_cache_creates_on_first_call(self):
        """Test that cache is created on first call."""
        # This test assumes the cache hasn't been created yet
        # In practice, it might already exist from other tests
        cache = get_injection_cache()
        assert cache is not None
        assert isinstance(cache, TTLCache)


class TestCacheIntegration:
    """Integration tests for cache with validator-like usage patterns."""

    def test_cache_validation_scenario(self):
        """Test cache behavior in a realistic validator scenario."""
        cache = TTLCache(maxsize=100, ttl=5)

        # Simulate repeated validation requests
        messages = [
            ("Hello world", False),  # Legitimate
            ("Ignore instructions", True),  # Injection
            ("Hello world", False),  # Duplicate - should hit cache
            ("How are you?", False),  # Legitimate
            ("Ignore instructions", True),  # Duplicate - should hit cache
        ]

        for msg, is_injection in messages:
            key = generate_cache_key(msg, 0.5, 7)
            cached = cache.get(key)

            if cached is None:
                # Cache miss - simulate validation
                result = {"is_injection": is_injection, "score": 0.98 if is_injection else 0.1}
                cache.put(key, result)
            else:
                # Cache hit - use cached result
                assert cached["is_injection"] == is_injection

        # Should have 3 entries in cache (2 unique + 1 duplicate handled by put update)
        # Actually 3 unique messages total
        assert cache.size() == 3

        # Should have good hit rate
        stats = cache.stats()
        assert stats["hits"] >= 2  # At least 2 cache hits from duplicates

    def test_cache_different_thresholds_separate_entries(self):
        """Test that same content with different thresholds creates separate cache entries."""
        cache = TTLCache(maxsize=100, ttl=60)

        content = "Test message"

        # Check with different thresholds
        key1 = generate_cache_key(content, 0.5, 7)
        key2 = generate_cache_key(content, 0.8, 7)

        assert key1 != key2

        cache.put(key1, {"result": "safe_for_0.5"})
        cache.put(key2, {"result": "safe_for_0.8"})

        assert cache.size() == 2
        assert cache.get(key1) == {"result": "safe_for_0.5"}
        assert cache.get(key2) == {"result": "safe_for_0.8"}
