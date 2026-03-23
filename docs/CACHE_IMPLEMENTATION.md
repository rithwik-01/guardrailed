# Prompt Injection Validator: LRU Cache Implementation

## Overview

This document describes the LRU (Least Recently Used) cache implementation for the Prompt Injection validator. The cache dramatically improves performance by avoiding redundant model inference for identical content.

**Performance Impact:**
- Cache hit: ~0.1ms (dict lookup)
- Cache miss: ~50-200ms (model inference)
- Target hit rate: >70% for typical workloads
- Expected latency reduction: 60-80% under typical load

## Architecture

### Cache Location
```
src/utils/cache.py          # Cache implementation
src/domain/validators/prompt_injection/main.py  # Cache usage
tests/unit/test_cache.py    # Cache unit tests
tests/unit/test_prompt_injection.py  # Integration tests
```

### Key Components

#### 1. TTLCache Class
A thread-safe, time-aware LRU cache with the following features:

- **TTL (Time-To-Live)**: Entries expire after a configured duration
- **LRU Eviction**: Least recently used entries are evicted when cache is full
- **Statistics**: Tracks hits, misses, evictions, and expirations
- **Thread Safety**: Safe for concurrent access in async environments

```python
from src.utils.cache import TTLCache

# Create a cache
cache = TTLCache(maxsize=1000, ttl=300, name="MyCache")

# Use the cache
cache.put("key", {"result": "value"})
result = cache.get("key")

# Get statistics
stats = cache.stats()
# {'hits': 10, 'misses': 2, 'evictions': 0, 'hit_rate': 0.8333}
```

#### 2. Cache Key Generation
Cache keys are generated using SHA-256 hashing to ensure:
- **Collision resistance**: Negligible chance of key collisions
- **Deterministic**: Same input always produces same key
- **Factors included**: Content, threshold, policy ID

```python
from src.utils.cache import generate_cache_key

key = generate_cache_key(
    content="Ignore instructions",
    threshold=0.5,
    policy_id=7
)
# Returns: 'a1b2c3d4e5f6...' (64-char hex string)
```

#### 3. Global Cache Singleton
A single cache instance is shared across all validator calls:

```python
from src.utils.cache import get_injection_cache

cache = get_injection_cache()
# Returns the global TTLCache instance for prompt injection
```

## Implementation Details

### Cache Entry Structure

Each cache entry contains:
```python
class CacheEntry:
    value: Any           # The cached Status object
    created_at: float    # Unix timestamp when created
    expires_at: float    # Unix timestamp when entry expires
```

### Cache Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    check_prompt_injection()                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │  Generate cache key    │
                 │  (content + threshold   │
                 │   + policy_id)          │
                 └────────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │   Check cache          │
                 │   cache.get(key)       │
                 └────────────────────────┘
                      │            │
           ┌──────────┘            └──────────┐
           │                                    │
      CACHE HIT                            CACHE MISS
           │                                    │
           ▼                                    ▼
  ┌──────────────────┐              ┌────────────────────┐
  │ Return cached    │              │ Run model           │
  │ Status object    │              │ inference           │
  │ (~0.1ms)         │              │ (~50-200ms)         │
  └──────────────────┘              └────────────────────┘
                                             │
                                             ▼
                                  ┌────────────────────┐
                                  │ Cache the result   │
                                  │ cache.put(key,     │
                                  │   result)          │
                                  └────────────────────┘
                                             │
                                             ▼
                                  ┌────────────────────┐
                                  │ Return Status      │
                                  │ object             │
                                  └────────────────────┘
```

### Cache Configuration

Default settings (defined in `get_injection_cache()`):
```python
TTLCache(
    maxsize=1000,    # Maximum 1000 entries
    ttl=300,         # 5 minutes expiration
    name="PromptInjectionCache"
)
```

**Why these values?**
- **maxsize=1000**: Sufficient for most workloads without excessive memory usage
- **ttl=300**: 5 minutes balances freshness with cache effectiveness
- **Memory estimate**: ~1000 entries × ~1KB per entry = ~1MB

### Thread Safety

The cache uses a threading.Lock to ensure thread-safe operations:
```python
def get(self, key: str) -> Optional[Any]:
    with self._lock:  # Ensures exclusive access
        entry = self._cache.get(key)
        # ... rest of logic
```

This is critical for async environments where multiple tasks may access the cache concurrently.

## Usage in Validator

### Integration Point

The cache is integrated into `check_prompt_injection()`:

```python
async def check_prompt_injection(message: str, policy: Policy) -> Tuple[Status, int]:
    # 1. Generate cache key
    cache = get_injection_cache()
    cache_key = generate_cache_key(
        content=message,
        threshold=policy.injection_threshold or 0.5,
        policy_id=policy.id,
    )

    # 2. Check cache
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        return cached_result, 0  # Cache hit!

    # 3. Cache miss - run model
    result = await loop.run_in_executor(None, _classifier, message)
    status_result = process_result(result, policy)

    # 4. Store in cache (unless error)
    if status_result.safety_code != SafetyCode.UNEXPECTED:
        cache.put(cache_key, status_result)

    return status_result, 0
```

### What Gets Cached

**Cached:**
- Safe results (`SafetyCode.SAFE`)
- Detected injections (`SafetyCode.INJECTION_DETECTED`)

**NOT Cached:**
- Error results (`SafetyCode.UNEXPECTED`) - might be transient
- Results with custom TTL (future enhancement)

## Testing

### Unit Tests

Located in `tests/unit/test_cache.py`:

```bash
# Run all cache tests
pytest tests/unit/test_cache.py -v

# Run specific test
pytest tests/unit/test_cache.py::test_cache_ttl_expiration -v
```

**Test Coverage:**
- Basic operations (get, put, clear)
- TTL expiration behavior
- LRU eviction when full
- Statistics tracking
- Cache key generation
- Thread safety
- Integration scenarios

### Integration Tests

Located in `tests/unit/test_prompt_injection.py`:

```bash
# Run cache integration tests
pytest tests/unit/test_prompt_injection.py -k "cache" -v
```

**Test Scenarios:**
- Cache hit on duplicate messages
- Cache miss on different content
- Separate cache entries for different thresholds
- Separate cache entries for different policy IDs
- Error results not cached
- Cache statistics logging

### Test Isolation

Each test automatically resets the cache via pytest fixture:

```python
@pytest.fixture(autouse=True)
def reset_cache_before_each_test():
    """Reset the injection cache before each test."""
    reset_injection_cache()
    yield
    reset_injection_cache()
```

## Monitoring & Observability

### Cache Statistics

Access cache stats at runtime:

```python
from src.utils.cache import get_injection_cache

cache = get_injection_cache()
stats = cache.stats()

print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Size: {stats['size']}/{stats['maxsize']}")
print(f"Hits: {stats['hits']}, Misses: {stats['misses']}")
print(f"Evictions: {stats['evictions']}, Expirations: {stats['expirations']}")
```

### Logging

The cache logs debug-level messages:

```python
logger.debug(
    f"Prompt injection cache HIT for policy {policy.id} "
    f"(threshold: {threshold}, cache stats: {cache.stats()})"
)
```

**Example log output:**
```
DEBUG: Prompt injection cache MISS for policy 7 (threshold: 0.5, cache stats: {'hits': 45, 'misses': 12, 'evictions': 2, 'size': 23, 'maxsize': 1000, 'hit_rate': 0.7895})
DEBUG: Prompt injection cache HIT for policy 7 (threshold: 0.5, cache stats: {'hits': 46, 'misses': 12, 'evictions': 2, 'size': 23, 'maxsize': 1000, 'hit_rate': 0.7931})
```

### Recommended Metrics to Track

1. **Hit Rate**: Target >70%
   - Low hit rate: Consider increasing TTL or maxsize
   - Very high hit rate (>95%): May indicate TTL is too long

2. **Evictions**: Track rate of evictions
   - High evictions: Increase maxsize or decrease TTL
   - Zero evictions: Cache might be oversized

3. **Cache Size**: Monitor current size vs maxsize
   - Consistently at maxsize: Consider increasing maxsize
   - Much smaller than maxsize: Could reduce maxsize to save memory

## Configuration & Tuning

### Adjusting Cache Size

To change maxsize, modify `get_injection_cache()` in `src/utils/cache.py`:

```python
_injection_cache = TTLCache(
    maxsize=2000,  # Increased from 1000
    ttl=300,
    name="PromptInjectionCache",
)
```

**When to increase maxsize:**
- High eviction rate
- Low hit rate with high request volume
- Sufficient memory available

**When to decrease maxsize:**
- Memory constraints
- Low request volume
- Cache never fills up

### Adjusting TTL

To change TTL, modify `get_injection_cache()`:

```python
_injection_cache = TTLCache(
    maxsize=1000,
    ttl=600,  # Increased from 300 to 600 seconds (10 minutes)
    name="PromptInjectionCache",
)
```

**Longer TTL pros/cons:**
- ✅ Higher hit rate
- ✅ Fewer model inference calls
- ❌ Stale results if models/behaviors change
- ❌ More memory usage

**Shorter TTL pros/cons:**
- ✅ Fresher results
- ✅ Faster adaptation to changes
- ❌ Lower hit rate
- ❌ More model inference calls

### Making Cache Configurable

For production, consider making cache settings configurable via environment variables or config file:

```python
import os

maxsize = int(os.getenv("INJECTION_CACHE_MAXSIZE", "1000"))
ttl = int(os.getenv("INJECTION_CACHE_TTL", "300"))

_injection_cache = TTLCache(
    maxsize=maxsize,
    ttl=ttl,
    name="PromptInjectionCache",
)
```

## Future Enhancements

### Phase 2.2: Metrics Collection
- Export stats to Prometheus/Grafana
- Track inference time percentiles (p50, p95, p99)
- Alert on low hit rates

### Phase 2.3: Batch Processing
- Support multiple messages in one call
- Leverage model's batch capability
- Improve throughput for bulk validation

### Distributed Caching
- Replace in-memory cache with Redis
- Share cache across multiple instances
- Improve hit rate in distributed deployments

### Smart Eviction
- Implement size-aware eviction (memory-based)
- Prioritize high-value entries
- Machine learning-based cache size optimization

### Cache Warming
- Pre-populate cache with common patterns
- Warm on application startup
- Reduce cold-start misses

## Troubleshooting

### Problem: Low Hit Rate (<50%)

**Possible Causes:**
1. High content variety (few duplicates)
2. TTL too short (entries expire before reuse)
3. Cache too small (frequent evictions)

**Solutions:**
- Increase TTL: `ttl=600` (10 minutes)
- Increase maxsize: `maxsize=2000`
- Monitor content diversity

### Problem: High Eviction Rate

**Possible Causes:**
1. Cache too small for workload
2. High request volume with many unique entries

**Solutions:**
- Increase maxsize
- Monitor memory usage
- Consider distributed caching

### Problem: High Memory Usage

**Possible Causes:**
1. maxsize too large
2. Cached Status objects are large

**Solutions:**
- Decrease maxsize
- Reduce TTL
- Implement size-based eviction

### Problem: Stale Results

**Possible Causes:**
1. TTL too long
2. Model behavior changed but cache not cleared

**Solutions:**
- Reduce TTL
- Implement cache invalidation on model updates
- Add `reset_injection_cache()` to deployment process

## Performance Benchmarks

### Expected Performance (1000 requests)

| Scenario | Avg Latency | Total Time | Cache Hit Rate |
|----------|-------------|------------|----------------|
| No cache | 100ms | 100s | 0% |
| 50% hit rate | 50ms | 50s | 50% |
| 70% hit rate | 30ms | 30s | 70% |
| 90% hit rate | 10ms | 10s | 90% |

### Memory Usage

Approximate memory per cache entry:
- CacheEntry overhead: ~100 bytes
- Status object: ~500 bytes
- Dict entry overhead: ~200 bytes
- **Total per entry**: ~800 bytes

For maxsize=1000: ~800 KB total
For maxsize=10000: ~8 MB total

## References

- **Implementation**: `src/utils/cache.py`
- **Usage**: `src/domain/validators/prompt_injection/main.py`
- **Tests**: `tests/unit/test_cache.py`
- **Integration Tests**: `tests/unit/test_prompt_injection.py` (cache_* tests)

## Summary

The LRU cache implementation provides:
- ✅ Significant performance improvement (60-80% latency reduction)
- ✅ Thread-safe operation for async environments
- ✅ Configurable size and TTL
- ✅ Comprehensive statistics for monitoring
- ✅ Full test coverage
- ✅ Well-documented for future maintenance

The cache is production-ready and can be further enhanced with distributed caching, metrics export, and smart eviction strategies as described in the Future Enhancements section.
