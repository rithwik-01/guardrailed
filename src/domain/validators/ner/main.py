import logging
from typing import Any, Dict, List, Optional

from fastapi import status

from src.core.state import app_state
from src.exceptions import NotInitializedError
from src.shared import Policy, Result, SafetyCode, Status

logger = logging.getLogger(__name__)


async def _get_ner_results(
    message_text: str, ner_results: Optional[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Helper to get NER results, either pre-computed or by prediction."""
    if ner_results is not None:
        return ner_results
    else:
        model = app_state.ner_model
        if model is None:
            logger.error("NER model not initialized during check")
            raise NotInitializedError("NER model")
        try:
            results, _ = await model.predict(message_text)
            return results
        except Exception as e:
            logger.error(f"Error during NER prediction: {e}", exc_info=True)
            return []


async def check_competitors(
    message_text: str,
    policy: Policy,
    ner_results: Optional[List[Dict[str, Any]]] = None,
) -> Status:
    try:
        competitors = getattr(policy, "competitors", [])
        threshold = getattr(policy, "threshold", 0.5)
        policy_message = getattr(policy, "message", "Competitor detected.")

        if not competitors:
            return Result.safe_result()

        results = await _get_ner_results(message_text, ner_results)

        if not results:
            return Result.safe_result()

        detected = []
        for result in results:
            entity_group = result.get("entity_group")
            score = result.get("score", 0.0)
            word = result.get("word", "")
            if score >= threshold and entity_group in ["B-ORG", "I-ORG", "ORG"]:
                if any(comp.lower() in word.lower() for comp in competitors):
                    detected.append(word)

        if detected:
            logger.info(
                f"Competitor detected: {detected}", extra={"policy_id": policy.id}
            )
            return Status(
                safety_code=SafetyCode.COMPETITOR_DETECTED,
                message=policy_message,
                status=status.HTTP_200_OK,
                action=policy.action,
            )
        return Result.safe_result()
    except NotInitializedError:
        logger.error(
            "Competitor check failed: NER model not initialized.", exc_info=True
        )
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error during competitor check processing: {e}", exc_info=True
        )
        return Result.safe_result()


async def check_persons(
    message_text: str,
    policy: Policy,
    ner_results: Optional[List[Dict[str, Any]]] = None,
) -> Status:
    try:
        persons = getattr(policy, "persons", [])
        threshold = getattr(policy, "threshold", 0.5)
        policy_message = getattr(policy, "message", "Specified person detected.")
        if not persons:
            return Result.safe_result()

        results = await _get_ner_results(message_text, ner_results)
        if not results:
            return Result.safe_result()

        detected = []
        for result in results:
            entity_group = result.get("entity_group")
            score = result.get("score", 0.0)
            word = result.get("word", "")
            if score >= threshold and entity_group in ["B-PER", "I-PER", "PER"]:
                if any(person.lower() in word.lower() for person in persons):
                    detected.append(word)

        if detected:
            logger.info(f"Person detected: {detected}", extra={"policy_id": policy.id})
            return Status(
                safety_code=SafetyCode.PERSON_DETECTED,
                message=policy_message,
                status=status.HTTP_200_OK,
                action=policy.action,
            )
        return Result.safe_result()
    except NotInitializedError:
        logger.error("Person check failed: NER model not initialized.", exc_info=True)
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error during person check processing: {e}", exc_info=True
        )
        return Result.safe_result()


async def check_locations(
    message_text: str,
    policy: Policy,
    ner_results: Optional[List[Dict[str, Any]]] = None,
) -> Status:
    try:
        locations = getattr(policy, "locations", [])
        threshold = getattr(policy, "threshold", 0.5)
        policy_message = getattr(policy, "message", "Specified location detected.")
        if not locations:
            return Result.safe_result()

        results = await _get_ner_results(message_text, ner_results)
        if not results:
            return Result.safe_result()

        detected = []
        for result in results:
            entity_group = result.get("entity_group")
            score = result.get("score", 0.0)
            word = result.get("word", "")
            if score >= threshold and entity_group in ["B-LOC", "I-LOC", "LOC"]:
                if any(loc.lower() in word.lower() for loc in locations):
                    detected.append(word)

        if detected:
            logger.info(
                f"Location detected: {detected}", extra={"policy_id": policy.id}
            )
            return Status(
                safety_code=SafetyCode.LOCATION_DETECTED,
                message=policy_message,
                status=status.HTTP_200_OK,
                action=policy.action,
            )
        return Result.safe_result()
    except NotInitializedError:
        logger.error("Location check failed: NER model not initialized.", exc_info=True)
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error during location check processing: {e}", exc_info=True
        )
        return Result.safe_result()
