import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import status

from src.core import app_state
from src.domain.validators.context import ValidationContext
from src.exceptions import NotInitializedError
from src.shared import Action, Agent, Policy, PolicyType, Result, SafetyCode, Status
from src.utils import chunk_text_by_char

from .ner import check_competitors, check_locations, check_persons
from .pii_leakage import check_pii
from .prompt_leakage import check_prompt
from .toxicity import check_toxicity
from .types import ContentMessage

logger = logging.getLogger(__name__)

assert app_state.config is not None, "app_state.config must be initialized"

ENABLE_CHUNKING = app_state.config.validation.enable_chunking
MAX_CHUNK_CHARS = app_state.config.validation.max_chunk_chars
CHUNK_OVERLAP_CHARS = app_state.config.validation.chunk_overlap_chars


class ContentValidator:
    """Validates content against locally configured policies."""

    def __init__(self, context: ValidationContext):
        self.context = context
        self.messages = context.messages
        self.policies = context.policies
        self.policy_check_functions = {
            PolicyType.PROFANITY: self._run_check_toxicity,
            PolicyType.COMPETITOR_CHECK: check_competitors,
            PolicyType.PERSON_CHECK: check_persons,
            PolicyType.LOCATION_CHECK: check_locations,
            PolicyType.PII_LEAKAGE: self._run_check_pii,
            PolicyType.PROMPT_LEAKAGE: self._run_check_prompt,
        }
        self.ner_policy_types = {
            PolicyType.COMPETITOR_CHECK,
            PolicyType.PERSON_CHECK,
            PolicyType.LOCATION_CHECK,
        }

    async def validate_content(self) -> Status:
        """Orchestrates validation, calling role validation."""
        start_time = time.time()
        request_id = getattr(asyncio.current_task(), "request_id", "unknown-validator")
        role_messages: Dict[str, List[Tuple[Dict, ContentMessage]]] = {}
        valid_messages_count = 0
        for raw_message in self.messages:
            role = raw_message.get("role")
            content = raw_message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                continue
            if role not in [Agent.USER, Agent.ASSISTANT, Agent.SYSTEM]:
                role = Agent.ASSISTANT
            if role not in role_messages:
                role_messages[role] = []
            content_message = ContentMessage(
                content=content, user_id=raw_message.get("user_id")
            )
            role_messages[role].append((raw_message, content_message))
            valid_messages_count += 1
        if valid_messages_count == 0:
            return Status(
                status=status.HTTP_400_BAD_REQUEST,
                message="No valid messages.",
                safety_code=SafetyCode.GENERIC_UNSAFE,
                action=Action.OVERRIDE.value,
            )
        validation_tasks = [
            self._validate_role_messages(role, messages)
            for role, messages in role_messages.items()
        ]
        role_results: List[Optional[Status]] = await asyncio.gather(*validation_tasks)
        first_unsafe_status = next(
            (
                res
                for res in role_results
                if isinstance(res, Status) and res.safety_code != SafetyCode.SAFE
            ),
            None,
        )
        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        if first_unsafe_status:
            action_name = (
                Action(first_unsafe_status.action).name
                if first_unsafe_status.action is not None
                else "None"
            )
            log_extra = {
                "request_id": request_id,
                "safety_code": first_unsafe_status.safety_code,
                "action": action_name,
                "elapsed_ms": elapsed_ms,
            }
            logger.info(
                f"Validation failed: {first_unsafe_status.message}", extra=log_extra
            )
            return first_unsafe_status
        else:
            logger.info(
                "All messages passed validation successfully",
                extra={
                    "request_id": request_id,
                    "message_count": valid_messages_count,
                    "elapsed_ms": elapsed_ms,
                },
            )
            return Result.safe_result()

    async def _run_checks_on_text(
        self,
        role: str,
        policy_groups: Dict[PolicyType, List[Policy]],
        ner_is_needed: bool,
        text_to_check: str,
        user_id: Optional[str],
        message_was_chunked: bool,
        request_id: str,
    ) -> Optional[Status]:
        """
        Runs applicable policy checks on the given text chunk or full message.
        Handles OBSERVE action centrally.
        Returns an unsafe Status object if a blocking violation occurs, None otherwise.
        """
        ner_results: Optional[List[Dict[str, Any]]] = None
        ner_error: bool = False
        log_extra_run = {
            "request_id": request_id,
            "role": role,
        }

        if ner_is_needed:
            if app_state.ner_model:
                try:
                    ner_results, _ = await app_state.ner_model.predict(text_to_check)
                    logger.debug("NER prediction successful", extra=log_extra_run)
                except Exception as e:
                    logger.error(
                        f"NER predict failed: {e}", extra=log_extra_run, exc_info=True
                    )
                    ner_error = True
            else:
                logger.error(
                    "NER model required by policies but not available.",
                    extra=log_extra_run,
                )
                return Result.unsafe_result(
                    "NER model required but not available.",
                    SafetyCode.UNEXPECTED,
                    action=Action.RETRY.value,
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                )

        check_tasks = []
        policy_task_map: Dict[asyncio.Task, Policy] = {}
        temp_content_message = ContentMessage(content=text_to_check, user_id=user_id)

        for policy_type, policies_of_type in policy_groups.items():
            if policy_type in self.ner_policy_types and (
                ner_error or not ner_is_needed
            ):
                if ner_error:
                    logger.warning(
                        f"Skipping NER policy type {policy_type.name} due to NER prediction error.",
                        extra=log_extra_run,
                    )
                continue

            handler = self.policy_check_functions.get(policy_type)
            if not handler:
                logger.warning(
                    f"No handler found for policy type {policy_type.name}. Skipping.",
                    extra=log_extra_run,
                )
                continue

            for policy in policies_of_type:
                if not policy.state:
                    continue

                try:
                    coro = None
                    if policy_type in self.ner_policy_types:
                        coro = handler(text_to_check, policy, ner_results=ner_results)  # type: ignore[operator]
                    elif policy_type == PolicyType.PII_LEAKAGE:
                        coro = self._run_check_pii(
                            temp_content_message, policy, message_was_chunked
                        )
                    elif policy_type in [
                        PolicyType.PROMPT_LEAKAGE,
                        PolicyType.PROFANITY,
                    ]:
                        coro = handler(temp_content_message, policy)  # type: ignore[operator]
                    else:
                        logger.error(
                            f"Unhandled policy type {policy_type.name} in task creation.",
                            extra=log_extra_run,
                        )
                        continue

                    if coro:
                        task = asyncio.create_task(coro)
                        check_tasks.append(task)
                        policy_task_map[task] = policy
                except Exception as task_creation_error:
                    logger.error(
                        f"Failed task creation for policy {policy.id} ({policy.name}): {task_creation_error}",
                        extra=log_extra_run,
                        exc_info=True,
                    )

        if not check_tasks:
            logger.debug(
                "No applicable check tasks created for this text.", extra=log_extra_run
            )
            return None

        logger.debug(f"Created {len(check_tasks)} check tasks.", extra=log_extra_run)

        first_blocking_status: Optional[Status] = None
        try:
            done, pending = await asyncio.wait(
                check_tasks, return_when=asyncio.ALL_COMPLETED
            )
            if pending:
                logger.warning(
                    f"{len(pending)} check tasks did not complete.", extra=log_extra_run
                )

            for task in done:
                if task not in policy_task_map:
                    logger.error(
                        f"Completed task {task} not found in policy map. Skipping.",
                        extra=log_extra_run,
                    )
                    continue

                policy = policy_task_map[task]

                try:
                    result_status: Status = await task

                    if result_status.safety_code != SafetyCode.SAFE:
                        if policy.action == Action.OBSERVE.value:
                            logger.info(
                                f"Observed violation for policy {policy.id} ({policy.name}). Message: '{result_status.message}'. Allowing request/chunk.",
                                extra=log_extra_run,
                            )
                        else:
                            logger.warning(
                                f"Blocking violation found for policy {policy.id} ({policy.name}). Action: {Action(policy.action).name}. Message: '{result_status.message}'.",
                                extra=log_extra_run,
                            )
                            final_status_code = result_status.status
                            if (
                                not (500 <= final_status_code < 600)
                                and final_status_code == status.HTTP_200_OK
                            ):
                                logger.warning(
                                    f"Check function for policy {policy.id} returned unsafe status code 200. Overriding to 400.",
                                    extra=log_extra_run,
                                )
                                final_status_code = status.HTTP_400_BAD_REQUEST

                            blocking_status = Status(
                                status=final_status_code,
                                message=result_status.message,
                                safety_code=result_status.safety_code,
                                action=policy.action,
                                processed_content=result_status.processed_content,
                            )

                            if first_blocking_status is None:
                                first_blocking_status = blocking_status

                except NotInitializedError as nie:
                    logger.error(
                        f"Component not ready for policy {policy.id} ({policy.name}): {nie}",
                        extra=log_extra_run,
                    )
                    blocking_status = Result.unsafe_result(
                        f"Service component not ready: {nie}",
                        SafetyCode.UNEXPECTED,
                        action=Action.RETRY.value,
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    )
                    if first_blocking_status is None:
                        first_blocking_status = blocking_status

                except Exception as task_exception:
                    logger.error(
                        f"Error executing check task for policy {policy.id} ({policy.name}): {task_exception}",
                        extra=log_extra_run,
                        exc_info=True,
                    )
                    blocking_status = Result.unsafe_result(
                        "Internal error during policy check.",
                        SafetyCode.UNEXPECTED,
                        action=Action.OVERRIDE.value,
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    )
                    if first_blocking_status is None:
                        first_blocking_status = blocking_status

        except Exception as wait_exception:
            logger.error(
                f"Error waiting for check tasks: {wait_exception}",
                extra=log_extra_run,
                exc_info=True,
            )
            return Result.unsafe_result(
                "Internal error during validation task processing.",
                SafetyCode.UNEXPECTED,
                action=Action.OVERRIDE.value,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        if first_blocking_status:
            logger.debug(
                f"Returning first blocking status for policy {policy.id}: {first_blocking_status.safety_code}",
                extra=log_extra_run,
            )
        else:
            logger.debug(
                "All checks passed or resulted in OBSERVE action.", extra=log_extra_run
            )

        return first_blocking_status

    async def _validate_role_messages(
        self, role: str, message_pairs: List[Tuple[Dict, ContentMessage]]
    ) -> Optional[Status]:
        """
        Validates messages for a specific role. Uses chunking if enabled and needed.
        Stops processing and returns the status immediately upon finding the first violation
        for any message within this role.
        """
        request_id = getattr(
            asyncio.current_task(), "request_id", f"unknown-validator-{role}"
        )
        log_extra_base = {"request_id": request_id, "role": role}

        active_policies = self._get_active_policies_for_role(role)
        if not active_policies:
            logger.debug(
                f"No active policies found for role '{role}'. Skipping validation for this role.",
                extra=log_extra_base,
            )
            return None

        policy_groups: Dict[PolicyType, List[Policy]] = {}
        for policy in active_policies:
            try:
                ptype = PolicyType(policy.id)
                policy_groups.setdefault(ptype, []).append(policy)
            except ValueError:
                logger.warning(
                    f"Invalid policy ID {policy.id} encountered for role '{role}'. Skipping this policy.",
                    extra=log_extra_base,
                )
                continue
        if not policy_groups:
            logger.debug(
                f"No valid/applicable policies found for role '{role}' after grouping.",
                extra=log_extra_base,
            )
            return None

        ner_is_needed = any(pt in self.ner_policy_types for pt in policy_groups)
        logger.debug(
            f"NER needed for role '{role}': {ner_is_needed}", extra=log_extra_base
        )

        for _raw_message, content_message in message_pairs:
            original_content = content_message.content
            message_was_chunked = False
            final_status_for_message: Optional[Status] = None
            message_log_extra = {
                **log_extra_base,
                "content_length": len(original_content),
            }

            should_chunk = ENABLE_CHUNKING and len(original_content) > MAX_CHUNK_CHARS

            if should_chunk:
                message_was_chunked = True
                text_chunks = chunk_text_by_char(
                    original_content, MAX_CHUNK_CHARS, CHUNK_OVERLAP_CHARS
                )
                logger.info(
                    f"Processing long message in {len(text_chunks)} chunks.",
                    extra=message_log_extra,
                )

                for chunk_index, (chunk_text, _) in enumerate(text_chunks):
                    chunk_log_extra = {
                        **message_log_extra,
                        "chunk_index": chunk_index + 1,
                        "total_chunks": len(text_chunks),
                    }
                    logger.debug(
                        f"Processing chunk {chunk_index+1}/{len(text_chunks)}.",
                        extra=chunk_log_extra,
                    )

                    chunk_status = await self._run_checks_on_text(
                        role=role,
                        policy_groups=policy_groups,
                        ner_is_needed=ner_is_needed,
                        text_to_check=chunk_text,
                        user_id=content_message.user_id,
                        message_was_chunked=message_was_chunked,
                        request_id=request_id,
                    )

                    if chunk_status is not None:
                        final_status_for_message = chunk_status
                        action_name = (
                            Action(chunk_status.action).name
                            if chunk_status.action is not None
                            else "None"
                        )
                        logger.info(
                            f"Violation found in chunk {chunk_index+1}. Policy Msg: '{chunk_status.message}'",
                            extra={
                                **chunk_log_extra,
                                "safety_code": chunk_status.safety_code,
                                "action": action_name,
                            },
                        )
                        break

            else:
                logger.debug(
                    "Processing message without chunking.", extra=message_log_extra
                )
                final_status_for_message = await self._run_checks_on_text(
                    role=role,
                    policy_groups=policy_groups,
                    ner_is_needed=ner_is_needed,
                    text_to_check=original_content,
                    user_id=content_message.user_id,
                    message_was_chunked=message_was_chunked,
                    request_id=request_id,
                )
                if final_status_for_message is not None:
                    action_name = (
                        Action(final_status_for_message.action).name
                        if final_status_for_message.action is not None
                        else "None"
                    )
                    logger.info(
                        f"Violation found in full message. Policy Msg: '{final_status_for_message.message}'",
                        extra={
                            **message_log_extra,
                            "safety_code": final_status_for_message.safety_code,
                            "action": action_name,
                        },
                    )

            if final_status_for_message is not None:
                logger.debug(
                    f"Returning early for role '{role}' due to violation.",
                    extra=log_extra_base,
                )
                return final_status_for_message

            logger.debug("Message passed all checks.", extra=message_log_extra)

        logger.debug(
            f"All messages for role '{role}' passed validation.", extra=log_extra_base
        )
        return None

    async def _run_check_pii(
        self, message: ContentMessage, policy: Policy, message_was_chunked: bool = False
    ) -> Status:
        """Async wrapper for PII check, aware of chunking for redaction."""
        try:
            result: Status = await check_pii(message.content, policy)
            if (
                message_was_chunked
                and result.safety_code != SafetyCode.SAFE
                and result.action == Action.REDACT.value
            ):
                if result.processed_content is not None:
                    req_id = getattr(
                        asyncio.current_task(), "request_id", "unknown-pii-wrapper"
                    )
                    logger.warning(
                        f"PII Redact on chunked message ({policy.id}). Clearing processed_content.",
                        extra={"request_id": req_id},
                    )
                    result.processed_content = None
            return result
        except NotInitializedError:
            raise
        except Exception as e:
            logger.error(f"Error in PII check wrapper: {e}", exc_info=True)
            return Result.unsafe_result(
                "Error during PII check",
                SafetyCode.UNEXPECTED,
                action=policy.action,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    async def _run_check_toxicity(
        self, message: ContentMessage, policy: Policy
    ) -> Status:
        """Async wrapper for toxicity check."""
        try:
            status_result, _ = await check_toxicity(message.content, policy)
            if (
                status_result.safety_code == SafetyCode.SAFE
                and policy.action == Action.OBSERVE.value
            ):
                status_result.action = Action.OBSERVE.value
            elif (
                status_result.safety_code != SafetyCode.SAFE
                and status_result.action is None
            ):
                status_result.action = policy.action
            return status_result
        except Exception as e:
            logger.error(f"Error in toxicity check wrapper: {e}", exc_info=True)
            return Result.unsafe_result(
                "Error during toxicity check",
                SafetyCode.UNEXPECTED,
                action=policy.action,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    async def _run_check_prompt(
        self, message: ContentMessage, policy: Policy
    ) -> Status:
        """Async wrapper for prompt leakage check (assuming check_prompt is sync)."""
        try:
            result: Status = check_prompt(message.content, policy)
            if result.action is None:
                result.action = policy.action
            return result
        except Exception as e:
            logger.error(f"Error in prompt leakage check wrapper: {e}", exc_info=True)
            return Result.unsafe_result(
                "Error during prompt leakage check",
                SafetyCode.UNEXPECTED,
                action=policy.action,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def _get_active_policies_for_role(self, role: str) -> List[Policy]:
        """Filters policies for the given role."""
        if not self.policies:
            return []
        is_user = role != Agent.ASSISTANT
        is_llm = role == Agent.ASSISTANT
        return [
            p
            for p in self.policies
            if isinstance(p, Policy)
            and p.state
            and ((p.is_llm_policy and is_llm) or (p.is_user_policy and is_user))
        ]
