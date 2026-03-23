import asyncio
import logging
from typing import Tuple

import torch
from transformers import pipeline

from fastapi import status

from src.shared import Action, Policy, Result, SafetyCode, Status

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

    Args:
        message: The text content to check.
        policy: The specific prompt injection policy being applied.

    Returns:
        A tuple containing:
            - Status: Indicates SAFE or INJECTION_DETECTED, including action and message.
            - int: The token count of the processed message.
    """
    policy_message = getattr(policy, "message", "Prompt injection detected.")
    threshold = 0.5
    if policy.injection_threshold is not None:
        threshold = policy.injection_threshold

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
                return (
                    Result.unsafe_result(
                        message=policy_message,
                        safety_code=SafetyCode.INJECTION_DETECTED,
                        action=policy.action,
                    ),
                    0,  # Token count not needed for this validator
                )
        # If we get here, either no injection or below threshold
        return Result.safe_result(), 0

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
            0,
        )
