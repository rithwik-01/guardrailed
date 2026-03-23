import asyncio
import logging
from typing import Tuple

import torch
from transformers import pipeline

from fastapi import status

from src.shared import Action, Policy, Result, SafetyCode, Status
from src.utils import generate_cache_key, get_injection_cache

logger = logging.getLogger(__name__)

# Load the model as a module-level singleton
_classifier = pipeline(
    "text-classification",
    model="deepset/deberta-v3-base-injection",
    truncation=True,
    max_length=512,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)


async def check_prompt_injection(message: str, policy: Policy) -> Tuple[Status, int]:
    """
    Checks message content for prompt injection attempts.

    This function implements a caching layer to avoid redundant model inference
    for identical content. Cache keys are generated based on message content,
    threshold, and policy ID to ensure correctness.

    Cache Behavior:
        - Hit: Returns cached result without model inference (fast path)
        - Miss: Runs model inference and caches result (slow path)
        - TTL: Cached entries expire after 5 minutes (configurable)
        - Size: Cache holds up to 1000 entries (configurable)

    Performance Characteristics:
        - Cache hit: ~0.1ms (dict lookup)
        - Cache miss: ~50-200ms (model inference)
        - Target hit rate: >70% for typical workloads

    Args:
        message: The text content to check.
        policy: The specific prompt injection policy being applied.

    Returns:
        A tuple containing:
            - Status: Indicates SAFE or INJECTION_DETECTED, including action and message.
            - int: The token count of the processed message (always 0 for this validator).
    """
    policy_message = getattr(policy, "message", "Prompt injection detected.")
    threshold = 0.5
    if policy.injection_threshold is not None:
        threshold = policy.injection_threshold

    # CHECK CACHE FIRST
    # Generate a cache key based on content, threshold, and policy ID
    # This ensures that different thresholds or policies don't share cache entries
    cache = get_injection_cache()
    cache_key = generate_cache_key(
        content=message,
        threshold=threshold,
        policy_id=policy.id,
    )

    # Try to get cached result
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        # Cache hit - return cached Status object
        # The cached Status already contains the correct safety_code and action
        logger.debug(
            f"Prompt injection cache HIT for policy {policy.id} "
            f"(threshold: {threshold}, cache stats: {cache.stats()})"
        )
        return cached_result, 0

    # Cache miss - need to run model inference
    logger.debug(
        f"Prompt injection cache MISS for policy {policy.id} "
        f"(threshold: {threshold}, cache stats: {cache.stats()})"
    )

    try:
        # Run inference in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _classifier, message)

        # The pipeline returns a list of dicts like [{'label': 'INJECTION', 'score': 0.98}]
        if isinstance(result, list) and len(result) > 0:
            prediction = result[0]
            label = prediction.get("label", "")
            score = prediction.get("score", 0.0)

            is_injection = label == "INJECTION" and score >= threshold

            if is_injection:
                logger.warning(
                    f"Prompt injection detected with score {score:.4f} (threshold: {threshold}) for policy {policy.id}. Action: {Action(policy.action).name}"
                )
                status_result = Result.unsafe_result(
                    message=policy_message,
                    safety_code=SafetyCode.INJECTION_DETECTED,
                    action=policy.action,
                )
                # Cache the result
                cache.put(cache_key, status_result)
                return status_result, 0

        # If we get here, either no injection or below threshold
        status_result = Result.safe_result()
        # Cache the safe result
        cache.put(cache_key, status_result)
        return status_result, 0

    except Exception as e:
        logger.error(
            f"Error during prompt injection check for policy {policy.id}: {e}",
            exc_info=True,
        )
        status_result = Result.unsafe_result(
            message="Internal error during prompt injection check.",
            safety_code=SafetyCode.UNEXPECTED,
            action=Action.OVERRIDE.value,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
        # Don't cache error results - they might be transient
        return status_result, 0
