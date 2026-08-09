import logging
from typing import Tuple

from fastapi import status

from src.core import app_state
from src.domain.transformers import ClassificationModel
from src.exceptions import NotInitializedError
from src.shared import Action, Policy, Result, SafetyCode, Status
from src.utils import generate_cache_key, get_injection_cache

logger = logging.getLogger(__name__)

# Label names used by the common injection classifiers for the positive class.
_INJECTION_LABEL_HINTS = ("inject", "jailbreak", "unsafe")
_DEFAULT_INJECTION_INDEX = 1
_injection_index_cache: dict[str, int] = {}


def _injection_index(model: ClassificationModel) -> int:
    """
    Resolve which softmax index carries the INJECTION probability.

    Different checkpoints label their classes differently (INJECTION/LEGIT,
    INJECTION/SAFE, LABEL_1/LABEL_0), so read it off the model config rather than
    assuming index 1 and silently inverting the guardrail on a model swap.
    """
    cached = _injection_index_cache.get(model.model_name)
    if cached is not None:
        return cached

    index = _DEFAULT_INJECTION_INDEX
    id2label = getattr(getattr(model.model, "config", None), "id2label", None)
    if isinstance(id2label, dict):
        for label_id, label in id2label.items():
            if isinstance(label, str) and any(
                hint in label.lower() for hint in _INJECTION_LABEL_HINTS
            ):
                index = int(label_id)
                break
        else:
            logger.warning(
                f"No injection-like label found in {model.model_name} "
                f"({id2label}). Falling back to index {index}."
            )

    _injection_index_cache[model.model_name] = index
    return index


async def check_prompt_injection(message: str, policy: Policy) -> Tuple[Status, int]:
    """
    Checks message content for prompt injection attempts.

    The model score is cached per (model, content) so that repeated content skips
    inference entirely. The policy threshold is applied *after* the cache lookup,
    which means policies with different thresholds share one cache entry and a
    threshold change takes effect immediately instead of waiting out the TTL.

    Args:
        message: The text content to check.
        policy: The specific prompt injection policy being applied.

    Returns:
        A tuple containing:
            - Status: Indicates SAFE or INJECTION_DETECTED, including action and message.
            - int: The token count of the processed message (0 on a cache hit).

    Raises:
        NotInitializedError: If the injection model is not available.
    """
    policy_message = getattr(policy, "message", "Prompt injection detected.")
    threshold = (
        policy.injection_threshold if policy.injection_threshold is not None else 0.5
    )

    model = app_state.injection_model
    if model is None:
        logger.error("Injection model not initialized during check")
        raise NotInitializedError("Prompt injection model")

    cache = get_injection_cache()
    cache_key = generate_cache_key(content=message, extra={"model": model.model_name})
    token_count = 0

    try:
        score = cache.get(cache_key)
        if score is None:
            probabilities, token_count = await model.predict(message)
            score = probabilities[_injection_index(model)]
            cache.put(cache_key, score)
            logger.debug(
                f"Prompt injection cache MISS for policy {policy.id} "
                f"(score: {score:.4f}, cache stats: {cache.stats()})"
            )
        else:
            logger.debug(
                f"Prompt injection cache HIT for policy {policy.id} "
                f"(score: {score:.4f}, cache stats: {cache.stats()})"
            )

        if score >= threshold:
            logger.warning(
                f"Prompt injection detected with score {score:.4f} "
                f"(threshold: {threshold}) for policy {policy.id}. "
                f"Action: {Action(policy.action).name}"
            )
            return (
                Result.unsafe_result(
                    message=policy_message,
                    safety_code=SafetyCode.INJECTION_DETECTED,
                    action=policy.action,
                ),
                token_count,
            )

        return Result.safe_result(), token_count

    except NotInitializedError:
        raise
    except Exception as e:
        logger.error(
            f"Error during prompt injection check for policy {policy.id}: {e}",
            exc_info=True,
        )
        return (
            Result.unsafe_result(
                message="Internal error during prompt injection check.",
                safety_code=SafetyCode.UNEXPECTED,
                action=Action.OVERRIDE.value,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            ),
            token_count,
        )
