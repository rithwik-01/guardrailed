import asyncio
import logging
from typing import List, Optional

from fastapi import status
from presidio_analyzer import RecognizerResult

from src.core import app_state
from src.exceptions import NotInitializedError
from src.shared import Action, Policy, Result, SafetyCode, Status

logger = logging.getLogger(__name__)

PresidioResultList = List[RecognizerResult]


def _get_entities_to_scan(policy: Policy) -> Optional[List[str]]:
    """Determines the final list of entities to scan based on policy."""
    entities_to_scan: Optional[List[str]] = None
    if policy.pii_entities:
        entities_to_scan = policy.pii_entities
        logger.debug(f"Scanning for specific entities: {entities_to_scan}")
    elif policy.pii_categories:
        logger.debug(
            f"Scanning using categories (via default recognizers): {policy.pii_categories}"
        )
        entities_to_scan = None
    else:
        logger.warning(
            f"Policy {policy.id}: No specific entities or categories. Scanning default set."
        )
        entities_to_scan = None
    return entities_to_scan


async def check_pii(
    prompt: str,
    policy: Policy,
) -> Status:
    analyzer_engine = app_state.presidio_analyzer_engine
    if not analyzer_engine:
        logger.error("Presidio Analyzer engine not initialized during PII check.")
        raise NotInitializedError("Presidio Analyzer Engine")

    try:
        entities_to_scan = _get_entities_to_scan(policy)

        def analyze_sync() -> PresidioResultList:
            logger.debug(
                f"Running PII analysis (Policy: {policy.id}, Threshold: {policy.pii_threshold}, Entities: {entities_to_scan or 'Default'})"
            )
            return analyzer_engine.analyze(
                text=prompt,
                entities=entities_to_scan,
                language="en",
                score_threshold=policy.pii_threshold,
                return_decision_process=False,
            )

        final_analyzer_results: PresidioResultList = await asyncio.to_thread(
            analyze_sync
        )

        if not final_analyzer_results:
            logger.debug(f"No PII found for policy {policy.id}")
            return Result.safe_result()

        detected_entity_types = sorted(
            list(set(res.entity_type for res in final_analyzer_results))
        )
        log_message = (
            f"PII detected for policy {policy.id}. "
            f"Types: {detected_entity_types} (Count: {len(final_analyzer_results)})"
        )

        if policy.action == Action.OBSERVE.value:
            logger.info(f"{log_message}. Action=OBSERVE, allowing request to proceed.")
            return Result.safe_result()
        else:
            logger.warning(
                f"{log_message}. Action={Action(policy.action).name}, blocking request."
            )
            return Result.unsafe_result(
                message=policy.message,
                safety_code=SafetyCode.PII_DETECTED,
                action=policy.action,
                processed_content=None,
                status_code=status.HTTP_400_BAD_REQUEST,
            )

    except NotInitializedError:
        raise
    except Exception as e:
        logger.error(
            f"Error during PII check for policy {policy.id}: {e}", exc_info=True
        )
        return Result.unsafe_result(
            message="Internal error during PII check.",
            safety_code=SafetyCode.UNEXPECTED,
            action=Action.OVERRIDE.value,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
