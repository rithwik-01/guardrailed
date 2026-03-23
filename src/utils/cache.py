"""
TTL-based LRU Cache for Validator Results

This module provides a time-aware LRU (Least Recently Used) cache implementation
designed specifically for caching validator inference results. The cache combines
memory efficiency with automatic expiration of stale entries.

Key Features:
- TTL (Time-To-Live) support: Entries expire after a configured duration
- LRU eviction: Least recently used entries are evicted when cache is full
- Statistics tracking: Monitor cache hit/miss ratios and eviction rates
- Thread-safe: Safe to use in async/concurrent environments

Example Usage:
    >>> from src.utils.cache import TTLCache
    >>>
    >>> # Create a cache that stores up to 1000 entries for 5 minutes
    >>> cache = TTLCache(maxsize=1000, ttl=300)
    >>>
    >>> # Store a value
    >>> cache.put("key1", {"label": "INJECTION", "score": 0.98})
    >>>
    >>> # Retrieve a value
    >>> result = cache.get("key1")  # Returns the cached value
    >>>
    >>> # Check statistics
    >>> print(cache.stats())
    >>> {'hits': 10, 'misses': 2, 'evictions': 0, 'size': 1}

Architecture Notes:
- Uses OrderedDict for O(1) access and LRU ordering
- Lazy expiration: Entries are checked for expiration on access
- Maxsize enforcement: Oldest entries are evicted when limit is reached

Performance Characteristics:
- get(): O(1) average case
- put(): O(1) average case
- Memory: O(maxsize) - bounded by the maxsize parameter

Future Enhancements:
- Add Redis backend for distributed caching
- Add cache warming on startup
- Add cache invalidation hooks
- Add size-based eviction in addition to time-based
"""

import hashlib
import logging
import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CacheEntry:
    """
    Represents a single cache entry with value and expiration timestamp.

    Attributes:
        value: The cached value
        expires_at: Unix timestamp when this entry expires
        created_at: Unix timestamp when this entry was created
    """

    __slots__ = ["value", "expires_at", "created_at"]

    def __init__(self, value: Any, ttl_seconds: float):
        """
        Initialize a cache entry.

        Args:
            value: The value to cache
            ttl_seconds: Time-to-live in seconds
        """
        self.value = value
        self.created_at = time.time()
        self.expires_at = self.created_at + ttl_seconds

    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        return time.time() > self.expires_at

    def age_seconds(self) -> float:
        """Get the age of this entry in seconds."""
        return time.time() - self.created_at

    def ttl_remaining(self) -> float:
        """Get remaining TTL in seconds (0 if expired)."""
        remaining = self.expires_at - time.time()
        return max(0.0, remaining)


class TTLCache:
    """
    Thread-safe TTL-based LRU cache.

    This cache combines two eviction strategies:
    1. Time-based: Entries expire after TTL seconds
    2. Size-based: Least recently used entries are evicted when cache is full

    The cache is lazy with respect to expiration - entries are not proactively
    removed when they expire, but are checked and removed on access.

    Thread Safety:
        All public methods are thread-safe and can be called concurrently
        from multiple threads or async tasks.

    Example:
        >>> cache = TTLCache(maxsize=1000, ttl=300)
        >>> cache.put("user:123:prompt", {"score": 0.95})
        >>> result = cache.get("user:123:prompt")
        >>> print(result)  # {"score": 0.95}
    """

    def __init__(self, maxsize: int = 1000, ttl: int = 300, name: str = "TTLCache"):
        """
        Initialize the TTL cache.

        Args:
            maxsize: Maximum number of entries to store. When exceeded,
                    least recently used entries are evicted.
            ttl: Default time-to-live for entries in seconds
            name: Name for this cache instance (used in logging)

        Raises:
            ValueError: If maxsize <= 0 or ttl <= 0
        """
        if maxsize <= 0:
            raise ValueError(f"maxsize must be positive, got {maxsize}")
        if ttl <= 0:
            raise ValueError(f"ttl must be positive, got {ttl}")

        self.maxsize = maxsize
        self.default_ttl = ttl
        self.name = name

        # OrderedDict maintains insertion order for LRU tracking
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()

        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
        }

        logger.info(
            f"Initialized {self.name}: maxsize={maxsize}, ttl={ttl}s",
        )

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from the cache.

        If the key exists but the entry has expired, it will be removed
        and None will be returned.

        Args:
            key: The cache key to look up

        Returns:
            The cached value if found and not expired, None otherwise
        """
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats["misses"] += 1
                return None

            # Check expiration
            if entry.is_expired():
                # Remove expired entry
                del self._cache[key]
                self._stats["misses"] += 1
                self._stats["expirations"] += 1
                logger.debug(
                    f"{self.name}: Cache entry expired for key '{key[:50]}...'"
                )
                return None

            # Move to end (mark as recently used)
            self._cache.move_to_end(key)
            self._stats["hits"] += 1
            return entry.value

    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Store a value in the cache.

        If the key already exists, its value and TTL will be updated.
        If the cache is full, the least recently used entry will be evicted.

        Args:
            key: The cache key to store under
            value: The value to cache
            ttl: Optional custom TTL in seconds. If None, uses default TTL
        """
        if ttl is None:
            ttl = self.default_ttl

        with self._lock:
            # Check if we need to evict
            if key not in self._cache and len(self._cache) >= self.maxsize:
                # Evict oldest (first) entry
                oldest_key, oldest_entry = next(iter(self._cache.items()))
                del self._cache[oldest_key]
                self._stats["evictions"] += 1
                logger.debug(
                    f"{self.name}: Evicted entry '{oldest_key[:50]}...' "
                    f"(age: {oldest_entry.age_seconds():.1f}s)"
                )

            # Add or update entry
            entry = CacheEntry(value, ttl)
            self._cache[key] = entry

            # Move to end (mark as recently used)
            self._cache.move_to_end(key)

    def clear(self) -> None:
        """Clear all entries from the cache and reset statistics."""
        with self._lock:
            size = len(self._cache)
            self._cache.clear()
            self._stats = {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "expirations": 0,
            }
            logger.info(f"{self.name}: Cleared {size} entries")

    def size(self) -> int:
        """Get the current number of entries in the cache."""
        with self._lock:
            return len(self._cache)

    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache metrics:
            - hits: Number of successful cache lookups
            - misses: Number of cache misses (including expired)
            - evictions: Number of entries evicted due to size limit
            - expirations: Number of entries that expired on access
            - size: Current number of entries
            - maxsize: Maximum configured size
            - hit_rate: Ratio of hits to total lookups (0.0-1.0)
        """
        with self._lock:
            total_lookups = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_lookups if total_lookups > 0 else 0.0

            return {
                **self._stats,
                "size": len(self._cache),
                "maxsize": self.maxsize,
                "hit_rate": round(hit_rate, 4),
            }

    def cleanup_expired(self) -> int:
        """
        Manually trigger cleanup of all expired entries.

        This is not normally necessary as entries are lazily expired on access,
        but can be useful for maintenance or before getting stats.

        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = [
                key
                for key, entry in self._cache.items()
                if entry.is_expired()
            ]

            for key in expired_keys:
                del self._cache[key]
                self._stats["expirations"] += 1

            if expired_keys:
                logger.info(
                    f"{self.name}: Cleaned up {len(expired_keys)} expired entries"
                )

            return len(expired_keys)


def generate_cache_key(
    content: str,
    threshold: float,
    policy_id: int,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a deterministic cache key for validator results.

    The cache key incorporates all factors that affect the validation result:
    - Content hash (to detect identical input)
    - Threshold (different thresholds produce different results)
    - Policy ID (different policies may use different models)
    - Optional extra parameters for future extensibility

    Args:
        content: The input content being validated
        threshold: The validation threshold
        policy_id: The policy ID being used
        extra: Optional dict of additional parameters to include in key

    Returns:
        A hex string suitable for use as a cache key

    Example:
        >>> key = generate_cache_key("Hello world", 0.5, 7)
        >>> print(key)
        'a1b2c3d4e5f6...'

    Note:
        Uses SHA-256 for collision resistance. The probability of two
        different inputs producing the same hash is negligible (2^-256).
    """
    # Create a normalized representation of parameters
    key_parts = {
        "content": content,
        "threshold": threshold,
        "policy_id": policy_id,
    }

    if extra:
        # Sort keys for deterministic hashing
        for k in sorted(extra.keys()):
            key_parts[k] = extra[k]

    # Create a deterministic string representation
    key_str = str(key_parts)

    # Hash using SHA-256 for collision resistance
    hash_obj = hashlib.sha256(key_str.encode("utf-8"))
    return hash_obj.hexdigest()


# Global cache instance for prompt injection validator
# Initialized as None; set during application startup
_injection_cache: Optional[TTLCache] = None


def get_injection_cache() -> TTLCache:
    """
    Get or create the global prompt injection cache.

    This function provides a singleton cache instance that can be shared
    across all validator calls. Using a singleton ensures that cache entries
    are shared between requests, maximizing cache effectiveness.

    The cache is created with default settings on first call:
    - maxsize: 1000 entries
    - ttl: 300 seconds (5 minutes)

    Returns:
        The global TTLCache instance for prompt injection results

    Example:
        >>> from src.utils.cache import get_injection_cache
        >>>
        >>> cache = get_injection_cache()
        >>> cache.stats()
        >>> {'hits': 100, 'misses': 20, 'evictions': 5, ...}
    """
    global _injection_cache

    if _injection_cache is None:
        # Default configuration
        # TODO: Make these configurable via settings/config file
        _injection_cache = TTLCache(
            maxsize=1000,  # Store up to 1000 unique validation results
            ttl=300,  # Entries expire after 5 minutes
            name="PromptInjectionCache",
        )
        logger.info("Created global prompt injection cache")

    return _injection_cache


def reset_injection_cache() -> None:
    """
    Reset the global prompt injection cache.

    This is primarily useful for testing, but can also be used to
    manually clear the cache in production if needed.

    Example:
        >>> from src.utils.cache import reset_injection_cache
        >>>
        >>> reset_injection_cache()  # Clears all cached results
    """
    global _injection_cache

    if _injection_cache is not None:
        _injection_cache.clear()
        logger.info("Reset global prompt injection cache")
