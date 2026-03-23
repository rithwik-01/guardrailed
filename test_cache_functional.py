#!/usr/bin/env python3
"""
Quick functional test for the cache implementation.
This can be run independently to verify the cache works correctly.
"""

import sys
import time

# Add the src directory to the path
sys.path.insert(0, '/Users/reedula/Documents/palo/rebrand')

from src.utils.cache import (
    TTLCache,
    CacheEntry,
    generate_cache_key,
    get_injection_cache,
    reset_injection_cache,
)


def test_cache_basic_operations():
    """Test basic cache operations."""
    print("Testing basic cache operations...")
    cache = TTLCache(maxsize=10, ttl=60, name="TestCache")

    # Test put and get
    cache.put("key1", {"result": "INJECTION"})
    result = cache.get("key1")
    assert result == {"result": "INJECTION"}, f"Expected {{'result': 'INJECTION'}}, got {result}"
    print("  ✓ put and get work")

    # Test miss
    result = cache.get("nonexistent")
    assert result is None, f"Expected None for miss, got {result}"
    print("  ✓ cache miss returns None")

    # Test stats
    stats = cache.stats()
    assert stats["hits"] == 1, f"Expected 1 hit, got {stats['hits']}"
    assert stats["misses"] == 1, f"Expected 1 miss, got {stats['misses']}"
    print("  ✓ stats tracking works")


def test_cache_lru_eviction():
    """Test LRU eviction."""
    print("\nTesting LRU eviction...")
    cache = TTLCache(maxsize=3, ttl=60)

    cache.put("key1", "value1")
    cache.put("key2", "value2")
    cache.put("key3", "value3")
    cache.put("key4", "value4")  # Should evict key1

    assert cache.get("key1") is None, "key1 should have been evicted"
    assert cache.get("key2") == "value2", "key2 should still exist"
    print("  ✓ LRU eviction works")


def test_cache_ttl_expiration():
    """Test TTL expiration."""
    print("\nTesting TTL expiration...")
    cache = TTLCache(maxsize=10, ttl=1)  # 1 second TTL

    cache.put("key1", "value1")
    assert cache.get("key1") == "value1", "Should be available immediately"
    print("  ✓ Entry available immediately")

    time.sleep(1.5)
    result = cache.get("key1")
    assert result is None, f"Entry should have expired, got {result}"
    print("  ✓ TTL expiration works")


def test_cache_key_generation():
    """Test cache key generation."""
    print("\nTesting cache key generation...")

    key1 = generate_cache_key("test content", 0.5, 7)
    key2 = generate_cache_key("test content", 0.5, 7)
    assert key1 == key2, "Same inputs should produce same key"
    print("  ✓ Deterministic key generation")

    key3 = generate_cache_key("different content", 0.5, 7)
    assert key1 != key3, "Different content should produce different key"
    print("  ✓ Content-sensitive keys")

    key4 = generate_cache_key("test content", 0.8, 7)
    assert key1 != key4, "Different threshold should produce different key"
    print("  ✓ Threshold-sensitive keys")


def test_global_cache():
    """Test global cache singleton."""
    print("\nTesting global cache singleton...")
    reset_injection_cache()

    cache1 = get_injection_cache()
    cache2 = get_injection_cache()
    assert cache1 is cache2, "Should return same instance"
    print("  ✓ Singleton pattern works")

    cache1.put("test", "value")
    assert cache2.get("test") == "value", "Should share state"
    print("  ✓ State shared between instances")


def test_cache_entry():
    """Test CacheEntry class."""
    print("\nTesting CacheEntry...")
    entry = CacheEntry(value="test", ttl_seconds=10)

    assert entry.value == "test", "Value should be stored"
    assert not entry.is_expired(), "Should not be expired immediately"
    assert entry.ttl_remaining() > 0, "Should have remaining TTL"
    print("  ✓ CacheEntry creation works")

    # Test with very short TTL
    entry_short = CacheEntry(value="test", ttl_seconds=1)
    time.sleep(1.5)
    assert entry_short.is_expired(), "Should be expired after TTL"
    print("  ✓ CacheEntry expiration works")


def test_cache_clear():
    """Test cache clear."""
    print("\nTesting cache clear...")
    cache = TTLCache(maxsize=10, ttl=60)

    cache.put("key1", "value1")
    cache.put("key2", "value2")
    assert cache.size() == 2, "Should have 2 entries"

    cache.clear()
    assert cache.size() == 0, "Should be empty after clear"
    assert cache.get("key1") is None, "Entries should be gone"
    print("  ✓ Cache clear works")


def test_cache_stats():
    """Test cache statistics."""
    print("\nTesting cache statistics...")
    cache = TTLCache(maxsize=10, ttl=60)

    cache.put("key1", "value1")
    cache.get("key1")  # hit
    cache.get("key2")  # miss

    stats = cache.stats()
    assert stats["hits"] == 1, f"Expected 1 hit, got {stats['hits']}"
    assert stats["misses"] == 1, f"Expected 1 miss, got {stats['misses']}"
    assert stats["hit_rate"] == 0.5, f"Expected 0.5 hit rate, got {stats['hit_rate']}"
    print("  ✓ Statistics tracking works")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Cache Implementation Functional Test")
    print("=" * 60)

    try:
        test_cache_basic_operations()
        test_cache_lru_eviction()
        test_cache_ttl_expiration()
        test_cache_key_generation()
        test_global_cache()
        test_cache_entry()
        test_cache_clear()
        test_cache_stats()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
