from collections import defaultdict
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

import pytest
from fastapi import status
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from src.domain.validators.context import ValidationContext
from src.domain.validators.types import ContentMessage
from src.domain.validators.validate import ContentValidator
from src.exceptions import NotInitializedError
from src.shared import Action, Agent, Policy, PolicyType, Result, SafetyCode, Status

mock_ner_model_instance = AsyncMock(spec=True)
mock_profanity_model_instance = AsyncMock(spec=True)
mock_presidio_analyzer_instance = MagicMock(spec=AnalyzerEngine)
mock_presidio_anonymizer_instance = MagicMock(spec=AnonymizerEngine)

mock_ner_model_instance.predict = AsyncMock(return_value=([], 5))
mock_profanity_model_instance.predict = AsyncMock(return_value=((0.1, 0.9), 10))
mock_presidio_analyzer_instance.analyze = MagicMock(return_value=[])


@pytest.fixture
def sample_content_message():
    """Fixture for a ContentMessage instance."""
    return ContentMessage(content="Test message content.", user_id="user-wrap")


@pytest.fixture
def pii_policy_redact(sample_policy_pii):
    """Fixture for a PII policy with REDACT action."""
    policy = sample_policy_pii.model_copy()
    policy.action = Action.REDACT.value
    policy.message = "PII redacted."
    return policy


@pytest.fixture(scope="module", autouse=True)
def auto_mock_app_state():
    """Automatically patches app_state for all tests in this module."""
    mock_state = MagicMock()
    mock_state.ner_model = mock_ner_model_instance
    mock_state.profanity_model = mock_profanity_model_instance
    mock_state.presidio_analyzer_engine = mock_presidio_analyzer_instance
    mock_state.presidio_anonymizer_engine = mock_presidio_anonymizer_instance

    modules_to_patch = [
        "src.domain.validators.validate.app_state",
        "src.domain.validators.ner.main.app_state",
        "src.domain.validators.toxicity.main.app_state",
        "src.domain.validators.pii_leakage.main.app_state",
    ]
    patchers = [patch(module_path, mock_state) for module_path in modules_to_patch]

    try:
        for p in patchers:
            p.start()

        mock_ner_model_instance.reset_mock()
        mock_profanity_model_instance.reset_mock()
        mock_presidio_analyzer_instance.reset_mock()
        mock_presidio_anonymizer_instance.reset_mock()
        mock_ner_model_instance.predict.reset_mock(return_value=True, side_effect=True)
        mock_profanity_model_instance.predict.reset_mock(
            return_value=True, side_effect=True
        )
        mock_presidio_analyzer_instance.analyze.reset_mock(
            return_value=True, side_effect=True
        )
        mock_ner_model_instance.predict.return_value = ([], 5)
        mock_profanity_model_instance.predict.return_value = ((0.1, 0.9), 10)
        mock_presidio_analyzer_instance.analyze.return_value = []

        yield mock_state

    finally:
        for p in patchers:
            try:
                p.stop()
            except RuntimeError:
                pass


SAFE_STATUS = Result.safe_result()

PATCH_TARGET_WRAPPER_PII = (
    "src.domain.validators.validate.ContentValidator._run_check_pii"
)
PATCH_TARGET_WRAPPER_TOXICITY = (
    "src.domain.validators.validate.ContentValidator._run_check_toxicity"
)
PATCH_TARGET_WRAPPER_PROMPT = (
    "src.domain.validators.validate.ContentValidator._run_check_prompt"
)
PATCH_TARGET_CHECK_COMPETITORS = "src.domain.validators.validate.check_competitors"
PATCH_TARGET_CHECK_PERSONS = "src.domain.validators.validate.check_persons"
PATCH_TARGET_CHECK_LOCATIONS = "src.domain.validators.validate.check_locations"
PATCH_TARGET_CHUNK_FUNC = "src.domain.validators.validate.chunk_text_by_char"
PATCH_TARGET_ENABLE_CHUNKING = "src.domain.validators.validate.ENABLE_CHUNKING"


@pytest.fixture
def all_policies_os(
    sample_policy_pii: Policy,
    sample_policy_profanity: Policy,
    sample_policy_prompt_leakage: Policy,
    sample_policy_competitor: Policy,
    sample_policy_person: Policy,
    sample_policy_location: Policy,
):
    """Fixture providing a list of various active policies for OS core."""
    policies = [
        sample_policy_pii,
        sample_policy_profanity,
        sample_policy_prompt_leakage,
        sample_policy_competitor,
        sample_policy_person,
        sample_policy_location,
    ]
    for p in policies:
        p.state = True
        if p.id != PolicyType.PROMPT_LEAKAGE.value:
            p.is_user_policy = True
            p.is_llm_policy = True
        else:
            p.is_user_policy = False
            p.is_llm_policy = True
        p.metadata = getattr(p, "metadata", {})
        p.threshold = getattr(p, "threshold", None)
        p.locations = getattr(p, "locations", None)
        p.persons = getattr(p, "persons", None)
        p.competitors = getattr(p, "competitors", None)
        p.protected_prompts = getattr(p, "protected_prompts", None)
        p.pii_categories = getattr(p, "pii_categories", None)
        p.pii_entities = getattr(p, "pii_entities", None)
        p.pii_threshold = getattr(p, "pii_threshold", 0.5)
        p.prompt_leakage_threshold = getattr(p, "prompt_leakage_threshold", 0.85)
    return policies


@pytest.fixture
def simple_messages():
    """Standard safe user and assistant messages."""
    return [
        {"role": Agent.USER, "content": "Safe user message.", "user_id": "user-test"},
        {"role": Agent.ASSISTANT, "content": "Safe assistant reply."},
    ]


@pytest.fixture
def validation_context_os(all_policies_os: list[Policy], simple_messages: list[dict]):
    """Creates a ValidationContext suitable for the OS core."""
    return ValidationContext(
        policies=all_policies_os,
        messages=simple_messages,
        user_id="test-user-overall",
    )


class TestContentValidatorOS:
    @pytest.mark.parametrize(
        "role_to_test", [Agent.USER, Agent.ASSISTANT, Agent.SYSTEM]
    )
    @pytest.mark.asyncio
    async def test_get_active_policies_for_role_os(
        self,
        validation_context_os: ValidationContext,
        all_policies_os: list[Policy],
        role_to_test: str,
    ):
        validator = ContentValidator(validation_context_os)
        active_policies = validator._get_active_policies_for_role(role_to_test)
        active_policy_ids = {p.id for p in active_policies}
        expected_ids = {
            PolicyType.PII_LEAKAGE.value,
            PolicyType.PROFANITY.value,
            PolicyType.COMPETITOR_CHECK.value,
            PolicyType.PERSON_CHECK.value,
            PolicyType.LOCATION_CHECK.value,
        }
        if role_to_test == Agent.ASSISTANT:
            expected_ids.add(PolicyType.PROMPT_LEAKAGE.value)
        assert active_policy_ids == expected_ids

    @patch(
        "src.domain.validators.validate.ContentValidator._validate_role_messages",
        new_callable=AsyncMock,
    )
    @pytest.mark.asyncio
    async def test_validate_content_all_safe(
        self, mock_validate_role: AsyncMock, validation_context_os: ValidationContext
    ):
        """Test validation succeeds when all messages pass applicable policies."""
        mock_validate_role.return_value = None
        validator = ContentValidator(validation_context_os)

        result = await validator.validate_content()

        assert result.status == status.HTTP_200_OK
        assert result.safety_code == SafetyCode.SAFE
        assert result.message == "Message validated successfully"
        assert result.action is None

        assert mock_validate_role.call_count == 2
        user_call = next(
            (c for c in mock_validate_role.call_args_list if c.args[0] == Agent.USER),
            None,
        )
        asst_call = next(
            (
                c
                for c in mock_validate_role.call_args_list
                if c.args[0] == Agent.ASSISTANT
            ),
            None,
        )
        assert user_call is not None
        assert asst_call is not None

    @patch(
        "src.domain.validators.validate.ContentValidator._validate_role_messages",
        new_callable=AsyncMock,
    )
    @pytest.mark.asyncio
    async def test_validate_content_user_unsafe(
        self, mock_validate_role: AsyncMock, validation_context_os: ValidationContext
    ):
        """Test validation fails when a user message violates a policy."""
        unsafe_status = Status(
            status=status.HTTP_400_BAD_REQUEST,
            message="User message failed PII",
            safety_code=SafetyCode.PII_DETECTED,
            action=Action.OVERRIDE.value,
        )

        async def validate_role_side_effect(role, message_pairs):
            if role == Agent.USER:
                return unsafe_status
            return None

        mock_validate_role.side_effect = validate_role_side_effect
        validator = ContentValidator(validation_context_os)

        result = await validator.validate_content()

        assert result.status == unsafe_status.status
        assert result.safety_code == unsafe_status.safety_code
        assert result.message == unsafe_status.message
        assert result.action == unsafe_status.action

        user_call = next(
            (c for c in mock_validate_role.call_args_list if c.args[0] == Agent.USER),
            None,
        )
        assert user_call is not None

    @patch(
        "src.domain.validators.validate.ContentValidator._validate_role_messages",
        new_callable=AsyncMock,
    )
    @pytest.mark.asyncio
    async def test_validate_content_assistant_unsafe(
        self, mock_validate_role: AsyncMock, validation_context_os: ValidationContext
    ):
        """Test validation fails when an assistant message violates a policy."""
        unsafe_status = Status(
            status=status.HTTP_200_OK,
            message="Assistant leaked prompt",
            safety_code=SafetyCode.PROMPT_LEAKED,
            action=Action.OBSERVE.value,
        )

        async def validate_role_side_effect(role, message_pairs):
            if role == Agent.ASSISTANT:
                return unsafe_status
            return None

        mock_validate_role.side_effect = validate_role_side_effect
        validator = ContentValidator(validation_context_os)

        result = await validator.validate_content()

        assert result.status == unsafe_status.status
        assert result.safety_code == unsafe_status.safety_code
        assert result.message == unsafe_status.message
        assert result.action == unsafe_status.action
        assert mock_validate_role.call_count == 2

    @pytest.mark.asyncio
    async def test_validate_invalid_message_structure_os(
        self, validation_context_os: ValidationContext
    ):
        """Test validation returns 400 when *no* valid messages are present."""
        validation_context_os.messages = [
            {"content": "Missing role"},
            {"role": Agent.ASSISTANT},
        ]
        validator = ContentValidator(validation_context_os)
        result = await validator.validate_content()
        assert result.status == status.HTTP_400_BAD_REQUEST
        assert result.safety_code == SafetyCode.GENERIC_UNSAFE
        assert "No valid messages" in result.message
        assert result.action == Action.OVERRIDE.value

    @pytest.mark.asyncio
    async def test_validate_empty_messages_list_os(
        self, validation_context_os: ValidationContext
    ):
        """Test validation handles an empty messages list correctly (expect 400)."""
        validation_context_os.messages = []
        validator = ContentValidator(validation_context_os)
        result = await validator.validate_content()
        assert result.status == status.HTTP_400_BAD_REQUEST
        assert result.safety_code == SafetyCode.GENERIC_UNSAFE
        assert "No valid messages" in result.message
        assert result.action == Action.OVERRIDE.value

    @patch(PATCH_TARGET_WRAPPER_PII, new_callable=AsyncMock)
    @patch(PATCH_TARGET_WRAPPER_TOXICITY, new_callable=AsyncMock)
    @patch(PATCH_TARGET_WRAPPER_PROMPT, new_callable=AsyncMock)
    @patch(PATCH_TARGET_CHECK_COMPETITORS, new_callable=AsyncMock)
    @patch(PATCH_TARGET_CHECK_PERSONS, new_callable=AsyncMock)
    @patch(PATCH_TARGET_CHECK_LOCATIONS, new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_run_checks_on_text_calls_ner_when_needed(
        self,
        mock_check_loc: AsyncMock,
        mock_check_per: AsyncMock,
        mock_check_comp: AsyncMock,
        mock_wrapper_prompt: AsyncMock,
        mock_wrapper_tox: AsyncMock,
        mock_wrapper_pii: AsyncMock,
        validation_context_os: ValidationContext,
        auto_mock_app_state,
    ):
        """Verify NER model predict is called only when NER policies are active."""
        mock_wrapper_pii.return_value = SAFE_STATUS
        mock_wrapper_tox.return_value = SAFE_STATUS
        mock_wrapper_prompt.return_value = SAFE_STATUS
        mock_check_comp.return_value = SAFE_STATUS
        mock_check_per.return_value = SAFE_STATUS
        mock_check_loc.return_value = SAFE_STATUS

        validator = ContentValidator(validation_context_os)
        user_policies = validator._get_active_policies_for_role(Agent.USER)
        policy_groups = defaultdict(list)
        for policy in user_policies:
            try:
                ptype = PolicyType(policy.id)
                policy_groups[ptype].append(policy)
            except ValueError:
                continue

        test_text = "User text mentioning EvilCorp and Paris"
        ner_model_mock = auto_mock_app_state.ner_model
        ner_model_mock.predict.reset_mock()

        ner_is_needed_case1 = any(
            ptype in validator.ner_policy_types for ptype in policy_groups
        )
        assert ner_is_needed_case1 is True, "Fixture policies should require NER"

        result_case1 = await validator._run_checks_on_text(
            role=Agent.USER,
            policy_groups=policy_groups,
            ner_is_needed=ner_is_needed_case1,
            text_to_check=test_text,
            user_id="u1",
            message_was_chunked=False,
            request_id="req1",
        )

        assert result_case1 is None, f"Expected SAFE (None), got: {result_case1}"
        ner_model_mock.predict.assert_called_once_with(test_text)
        mock_check_comp.assert_called_once()
        mock_check_per.assert_called_once()
        mock_check_loc.assert_called_once()
        mock_wrapper_pii.assert_called_once()
        mock_wrapper_tox.assert_called_once()
        mock_wrapper_prompt.assert_not_called()

        ner_model_mock.predict.reset_mock()
        mock_check_comp.reset_mock()
        mock_check_per.reset_mock()
        mock_check_loc.reset_mock()
        mock_wrapper_pii.reset_mock()
        mock_wrapper_tox.reset_mock()
        mock_wrapper_prompt.reset_mock()

        ner_is_needed_case2 = False
        non_ner_policy_groups = {
            ptype: policies
            for ptype, policies in policy_groups.items()
            if ptype not in validator.ner_policy_types
        }
        assert non_ner_policy_groups

        result_case2 = await validator._run_checks_on_text(
            role=Agent.USER,
            policy_groups=non_ner_policy_groups,
            ner_is_needed=ner_is_needed_case2,
            text_to_check=test_text,
            user_id="u1",
            message_was_chunked=False,
            request_id="req2",
        )

        assert result_case2 is None, f"Expected SAFE (None), got: {result_case2}"
        ner_model_mock.predict.assert_not_called()
        mock_check_comp.assert_not_called()
        mock_check_per.assert_not_called()
        mock_check_loc.assert_not_called()
        mock_wrapper_pii.assert_called_once()
        mock_wrapper_tox.assert_called_once()
        mock_wrapper_prompt.assert_not_called()

    @patch(PATCH_TARGET_WRAPPER_PII, new_callable=AsyncMock)
    @patch(PATCH_TARGET_WRAPPER_TOXICITY, new_callable=AsyncMock)
    @patch(PATCH_TARGET_WRAPPER_PROMPT, new_callable=AsyncMock)
    @patch(PATCH_TARGET_CHECK_COMPETITORS, new_callable=AsyncMock)
    @patch(PATCH_TARGET_CHECK_PERSONS, new_callable=AsyncMock)
    @patch(PATCH_TARGET_CHECK_LOCATIONS, new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_run_checks_on_text_returns_first_failure(
        self,
        mock_check_loc: AsyncMock,
        mock_check_per: AsyncMock,
        mock_check_comp: AsyncMock,
        mock_wrapper_prompt: AsyncMock,
        mock_wrapper_tox: AsyncMock,
        mock_wrapper_pii: AsyncMock,
        validation_context_os: ValidationContext,
        auto_mock_app_state,
    ):
        """Verify _run_checks_on_text returns immediately on the first policy violation."""
        unsafe_tox_status = Status(
            status=400,
            message="Toxic!",
            safety_code=SafetyCode.PROFANE,
            action=Action.OVERRIDE.value,
        )
        mock_wrapper_tox.return_value = unsafe_tox_status
        mock_wrapper_pii.return_value = SAFE_STATUS
        mock_wrapper_prompt.return_value = SAFE_STATUS
        mock_check_comp.return_value = SAFE_STATUS
        mock_check_per.return_value = SAFE_STATUS
        mock_check_loc.return_value = SAFE_STATUS

        validator = ContentValidator(validation_context_os)
        user_policies = validator._get_active_policies_for_role(Agent.USER)
        policy_groups = defaultdict(list)
        for policy in user_policies:
            try:
                ptype = PolicyType(policy.id)
                policy_groups[ptype].append(policy)
            except ValueError:
                continue

        test_text = "Some user text that is toxic"
        ner_is_needed = any(
            ptype in validator.ner_policy_types for ptype in policy_groups
        )
        ner_model_mock = auto_mock_app_state.ner_model
        ner_model_mock.predict.reset_mock()

        result = await validator._run_checks_on_text(
            role=Agent.USER,
            policy_groups=policy_groups,
            ner_is_needed=ner_is_needed,
            text_to_check=test_text,
            user_id="u1",
            message_was_chunked=False,
            request_id="req3",
        )

        assert result == unsafe_tox_status
        mock_wrapper_tox.assert_called_once()
        if ner_is_needed:
            ner_model_mock.predict.assert_called_once_with(test_text)

    @patch(PATCH_TARGET_CHUNK_FUNC)
    @patch(
        "src.domain.validators.validate.ContentValidator._run_checks_on_text",
        new_callable=AsyncMock,
    )
    @pytest.mark.asyncio
    async def test_validate_role_messages_uses_chunking(
        self, mock_run_checks, mock_chunk_func, validation_context_os: ValidationContext
    ):
        """Verify that chunking is triggered for long messages if enabled."""
        long_content = "A" * 2000
        validation_context_long = ValidationContext(
            policies=validation_context_os.policies,
            messages=[{"role": Agent.USER, "content": long_content}],
            user_id=validation_context_os.user_id,
        )
        validator = ContentValidator(validation_context_long)
        chunks = [("chunk1_text", 0), ("chunk2_text", 1800)]
        mock_chunk_func.return_value = chunks
        mock_run_checks.return_value = None

        user_policies = validator._get_active_policies_for_role(Agent.USER)
        policy_groups = defaultdict(list)
        for p in user_policies:
            policy_groups[PolicyType(p.id)].append(p)

        with patch(PATCH_TARGET_ENABLE_CHUNKING, True):
            await validator._validate_role_messages(
                Agent.USER,
                [
                    (
                        validation_context_long.messages[0],
                        ContentMessage(content=long_content),
                    )
                ],
            )

        mock_chunk_func.assert_called_once_with(long_content, 1800, 200)
        assert mock_run_checks.call_count == len(chunks)
        mock_run_checks.assert_has_calls(
            [
                call(
                    role=Agent.USER,
                    policy_groups=policy_groups,
                    ner_is_needed=ANY,
                    text_to_check="chunk1_text",
                    user_id=ANY,
                    message_was_chunked=True,
                    request_id=ANY,
                ),
                call(
                    role=Agent.USER,
                    policy_groups=policy_groups,
                    ner_is_needed=ANY,
                    text_to_check="chunk2_text",
                    user_id=ANY,
                    message_was_chunked=True,
                    request_id=ANY,
                ),
            ],
            any_order=False,
        )

    @patch(PATCH_TARGET_CHUNK_FUNC)
    @patch(
        "src.domain.validators.validate.ContentValidator._run_checks_on_text",
        new_callable=AsyncMock,
    )
    @pytest.mark.asyncio
    async def test_validate_role_messages_no_chunking_short_message(
        self, mock_run_checks, mock_chunk_func, validation_context_os: ValidationContext
    ):
        """Verify chunking is NOT triggered for short messages."""
        validator = ContentValidator(validation_context_os)
        mock_run_checks.return_value = None
        message_data = validation_context_os.messages[0]
        message_pair = (
            message_data,
            ContentMessage(
                content=message_data["content"], user_id=message_data.get("user_id")
            ),
        )

        await validator._validate_role_messages(Agent.USER, [message_pair])

        mock_chunk_func.assert_not_called()
        mock_run_checks.assert_called_once()
        assert (
            mock_run_checks.call_args.kwargs["text_to_check"] == message_data["content"]
        )
        assert mock_run_checks.call_args.kwargs["message_was_chunked"] is False

    @patch(PATCH_TARGET_CHUNK_FUNC)
    @patch(
        "src.domain.validators.validate.ContentValidator._run_checks_on_text",
        new_callable=AsyncMock,
    )
    @pytest.mark.asyncio
    async def test_validate_role_messages_no_chunking_disabled(
        self, mock_run_checks, mock_chunk_func, validation_context_os: ValidationContext
    ):
        """Verify chunking is NOT triggered when ENABLE_CHUNKING is False."""
        long_content = "A" * 2000
        validation_context_long = ValidationContext(
            policies=validation_context_os.policies,
            messages=[{"role": Agent.USER, "content": long_content}],
            user_id=validation_context_os.user_id,
        )
        validator = ContentValidator(validation_context_long)
        mock_run_checks.return_value = None
        message_pair = (
            validation_context_long.messages[0],
            ContentMessage(content=long_content),
        )

        with patch(PATCH_TARGET_ENABLE_CHUNKING, False):
            await validator._validate_role_messages(Agent.USER, [message_pair])

        mock_chunk_func.assert_not_called()
        mock_run_checks.assert_called_once()
        assert mock_run_checks.call_args.kwargs["text_to_check"] == long_content
        assert mock_run_checks.call_args.kwargs["message_was_chunked"] is False

    @patch("src.domain.validators.validate.check_pii", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_run_check_pii_wrapper_success(
        self,
        mock_check_pii: AsyncMock,
        validation_context_os: ValidationContext,
        sample_content_message: ContentMessage,
        sample_policy_pii: Policy,
    ):
        """Test _run_check_pii wrapper successfully calls check_pii."""
        validator = ContentValidator(validation_context_os)
        expected_status = Result.safe_result()
        mock_check_pii.return_value = expected_status

        result = await validator._run_check_pii(
            sample_content_message, sample_policy_pii, message_was_chunked=False
        )

        assert result == expected_status
        mock_check_pii.assert_awaited_once_with(
            sample_content_message.content, sample_policy_pii
        )

    @patch("src.domain.validators.validate.check_pii", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_run_check_pii_wrapper_not_initialized(
        self,
        mock_check_pii: AsyncMock,
        validation_context_os: ValidationContext,
        sample_content_message: ContentMessage,
        sample_policy_pii: Policy,
    ):
        """Test _run_check_pii wrapper raises NotInitializedError."""
        validator = ContentValidator(validation_context_os)
        mock_check_pii.side_effect = NotInitializedError("Presidio Engine")

        with pytest.raises(NotInitializedError, match="Presidio Engine"):
            await validator._run_check_pii(sample_content_message, sample_policy_pii)
        mock_check_pii.assert_awaited_once()

    @patch("src.domain.validators.validate.check_pii", new_callable=AsyncMock)
    @patch("src.domain.validators.validate.logger")
    @pytest.mark.asyncio
    async def test_run_check_pii_wrapper_generic_exception(
        self,
        mock_logger: MagicMock,
        mock_check_pii: AsyncMock,
        validation_context_os: ValidationContext,
        sample_content_message: ContentMessage,
        sample_policy_pii: Policy,
    ):
        """Test _run_check_pii wrapper handles generic exceptions."""
        validator = ContentValidator(validation_context_os)
        error_msg = "PII check crashed"
        mock_check_pii.side_effect = Exception(error_msg)

        result = await validator._run_check_pii(
            sample_content_message, sample_policy_pii
        )

        assert result.safety_code == SafetyCode.UNEXPECTED
        assert result.status == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert result.action == sample_policy_pii.action
        assert "Error during PII check" in result.message
        mock_check_pii.assert_awaited_once()
        mock_logger.error.assert_called_once()
        assert "Error in PII check wrapper" in mock_logger.error.call_args[0][0]
        assert error_msg in mock_logger.error.call_args[0][0]

    @patch("src.domain.validators.validate.check_toxicity", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_run_check_toxicity_wrapper_success(
        self,
        mock_check_toxicity: AsyncMock,
        validation_context_os: ValidationContext,
        sample_content_message: ContentMessage,
        sample_policy_profanity: Policy,
    ):
        """Test _run_check_toxicity wrapper successfully calls check_toxicity."""
        validator = ContentValidator(validation_context_os)
        expected_status = Result.safe_result()
        mock_check_toxicity.return_value = (
            expected_status,
            10,
        )

        result = await validator._run_check_toxicity(
            sample_content_message, sample_policy_profanity
        )

        assert result == expected_status
        mock_check_toxicity.assert_awaited_once_with(
            sample_content_message.content, sample_policy_profanity
        )

    @patch("src.domain.validators.validate.check_toxicity", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_run_check_toxicity_wrapper_sets_action(
        self,
        mock_check_toxicity: AsyncMock,
        validation_context_os: ValidationContext,
        sample_content_message: ContentMessage,
        sample_policy_profanity: Policy,
    ):
        """Test _run_check_toxicity wrapper sets action correctly on unsafe result."""
        validator = ContentValidator(validation_context_os)
        unsafe_status = Result.unsafe_result(
            message="Unsafe",
            safety_code=SafetyCode.PROFANE,
            action=None,
        )
        mock_check_toxicity.return_value = (unsafe_status, 10)

        result = await validator._run_check_toxicity(
            sample_content_message, sample_policy_profanity
        )

        assert result.safety_code == SafetyCode.PROFANE
        assert result.action == sample_policy_profanity.action
        mock_check_toxicity.assert_awaited_once()

    @patch("src.domain.validators.validate.check_toxicity", new_callable=AsyncMock)
    @patch("src.domain.validators.validate.logger")
    @pytest.mark.asyncio
    async def test_run_check_toxicity_wrapper_exception(
        self,
        mock_logger: MagicMock,
        mock_check_toxicity: AsyncMock,
        validation_context_os: ValidationContext,
        sample_content_message: ContentMessage,
        sample_policy_profanity: Policy,
    ):
        """Test _run_check_toxicity wrapper handles exceptions."""
        validator = ContentValidator(validation_context_os)
        error_msg = "Toxicity check crashed"
        mock_check_toxicity.side_effect = Exception(error_msg)

        result = await validator._run_check_toxicity(
            sample_content_message, sample_policy_profanity
        )

        assert result.safety_code == SafetyCode.UNEXPECTED
        assert result.status == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert result.action == sample_policy_profanity.action
        assert "Error during toxicity check" in result.message
        mock_check_toxicity.assert_awaited_once()
        mock_logger.error.assert_called_once()
        assert "Error in toxicity check wrapper" in mock_logger.error.call_args[0][0]
        assert error_msg in mock_logger.error.call_args[0][0]

    @patch("src.domain.validators.validate.check_prompt")
    @pytest.mark.asyncio
    async def test_run_check_prompt_wrapper_success(
        self,
        mock_check_prompt: MagicMock,
        validation_context_os: ValidationContext,
        sample_content_message: ContentMessage,
        sample_policy_prompt_leakage: Policy,
    ):
        """Test _run_check_prompt wrapper successfully calls check_prompt."""
        validator = ContentValidator(validation_context_os)
        expected_status = Result.safe_result()
        mock_check_prompt.return_value = expected_status

        result = await validator._run_check_prompt(
            sample_content_message, sample_policy_prompt_leakage
        )

        assert result == expected_status
        mock_check_prompt.assert_called_once_with(
            sample_content_message.content, sample_policy_prompt_leakage
        )

    @patch("src.domain.validators.validate.check_prompt")
    @pytest.mark.asyncio
    async def test_run_check_prompt_wrapper_sets_action(
        self,
        mock_check_prompt: MagicMock,
        validation_context_os: ValidationContext,
        sample_content_message: ContentMessage,
        sample_policy_prompt_leakage: Policy,
    ):
        """Test _run_check_prompt wrapper sets action correctly on unsafe result."""
        validator = ContentValidator(validation_context_os)
        unsafe_status = Result.unsafe_result(
            message="Leaked", safety_code=SafetyCode.PROMPT_LEAKED, action=None
        )
        mock_check_prompt.return_value = unsafe_status

        result = await validator._run_check_prompt(
            sample_content_message, sample_policy_prompt_leakage
        )

        assert result.safety_code == SafetyCode.PROMPT_LEAKED
        assert result.action == sample_policy_prompt_leakage.action
        mock_check_prompt.assert_called_once()

    @patch("src.domain.validators.validate.check_prompt")
    @patch("src.domain.validators.validate.logger")
    @pytest.mark.asyncio
    async def test_run_check_prompt_wrapper_exception(
        self,
        mock_logger: MagicMock,
        mock_check_prompt: MagicMock,
        validation_context_os: ValidationContext,
        sample_content_message: ContentMessage,
        sample_policy_prompt_leakage: Policy,
    ):
        """Test _run_check_prompt wrapper handles exceptions."""
        validator = ContentValidator(validation_context_os)
        error_msg = "Prompt check crashed"
        mock_check_prompt.side_effect = Exception(error_msg)

        result = await validator._run_check_prompt(
            sample_content_message, sample_policy_prompt_leakage
        )

        assert result.safety_code == SafetyCode.UNEXPECTED
        assert result.status == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert result.action == sample_policy_prompt_leakage.action
        assert "Error during prompt leakage check" in result.message
        mock_check_prompt.assert_called_once()
        mock_logger.error.assert_called_once()
        assert (
            "Error in prompt leakage check wrapper" in mock_logger.error.call_args[0][0]
        )
        assert error_msg in mock_logger.error.call_args[0][0]
