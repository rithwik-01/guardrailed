import logging
from typing import Tuple

from fastapi import status

from src.core import app_state
from src.exceptions import NotInitializedError
from src.shared import Action, Policy, Result, SafetyCode, Status

logger = logging.getLogger(__name__)


async def check_toxicity(message: str, policy: Policy) -> Tuple[Status, int]:
    """
    Checks message content for toxicity based on the policy threshold.

    Args:
        message: The text content to check.
        policy: The specific toxicity policy being applied.

    Returns:
        A tuple containing:
            - Status: Indicates SAFE or PROFANE, including action and message.
            - int: The token count of the processed message.

    Raises:
        NotInitializedError: If the profanity model is not available.
    """
    message = message.lower()
    policy_message = getattr(policy, "message", "Toxicity detected.")
    try:
        model = app_state.profanity_model
        if model is None:
            logger.error("Profanity model not initialized during check")
            raise NotInitializedError("Profanity model")

        probabilities, token_count = await model.predict(message)

        threshold = 0.5
        if policy.threshold is not None:
            threshold = policy.threshold

        is_profanity = probabilities[1] > threshold

        if is_profanity:
            logger.warning(
                f"Profanity detected above threshold {threshold} (Score: {probabilities[1]:.4f}) for policy {policy.id}. Action: {Action(policy.action).name}"
            )
            return (
                Result.unsafe_result(
                    message=policy_message,
                    safety_code=SafetyCode.PROFANE,
                    action=policy.action,
                ),
                token_count,
            )
        result_status = Result.safe_result()
        return result_status, token_count

    except NotInitializedError:
        logger.error(
            f"Profanity check failed for policy {policy.id}: Model not initialized.",
            exc_info=False,
        )
        raise

    except Exception as e:
        logger.error(
            f"Error during profanity check for policy {policy.id}: {e}", exc_info=True
        )
        return (
            Result.unsafe_result(
                message="Internal error during profanity check.",
                safety_code=SafetyCode.UNEXPECTED,
                action=Action.OVERRIDE.value,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            ),
            0,
        )
