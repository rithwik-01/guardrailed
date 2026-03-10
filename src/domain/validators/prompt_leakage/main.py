import logging

from fastapi import status
from rapidfuzz import fuzz

from src.shared import Policy, Result, SafetyCode, Status
from src.utils.utils import normalize_text

logger = logging.getLogger(__name__)


def check_prompt(message: str, policy: Policy) -> Status:
    protected_prompts = getattr(policy, "protected_prompts", [])
    policy_message = getattr(policy, "message", "Prompt leakage detected.")
    threshold = getattr(policy, "prompt_leakage_threshold", 0.85) * 100

    if not protected_prompts:
        return Result.safe_result()

    normalized_message = normalize_text(message)
    if not normalized_message:
        return Result.safe_result()

    for prompt in protected_prompts:
        normalized_prompt = normalize_text(prompt)
        if not normalized_prompt:
            continue

        similarity_score = fuzz.partial_ratio(
            normalized_prompt, normalized_message, score_cutoff=threshold
        )

        if similarity_score >= threshold:
            logger.warning(
                f"Potential prompt leakage detected. Policy ID: {policy.id}. "
                f"Matched prompt (normalized): '{normalized_prompt[:50]}...'. "
                f"Similarity score: {similarity_score:.2f} >= threshold: {threshold}"
            )
            return Status(
                status=status.HTTP_200_OK,
                message=policy_message,
                safety_code=SafetyCode.PROMPT_LEAKED,
                action=policy.action,
            )

    return Result.safe_result()
