from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import yaml

from src.main import app
from src.presentation.dependencies import get_loaded_policies
from src.shared import Action, Policy, PolicyType


@pytest.fixture(scope="function")
def mock_http_client_instance_fixture():
    """Mocks the httpx.AsyncClient instance used within proxy_utils."""
    mock_client_instance = MagicMock(spec=httpx.AsyncClient)
    mock_client_instance.post = AsyncMock()
    mock_client_instance.aclose = AsyncMock(return_value=None)
    mock_client_instance.post.reset_mock(return_value=True, side_effect=True)

    with patch(
        "src.presentation.proxy_utils.http_client", mock_client_instance
    ) as mock_instance:
        yield mock_instance


@pytest.fixture(scope="function")
def mock_policies_dependency():
    """
    Mocks the get_loaded_policies dependency to return a predefined list
    for the duration of a test function.
    """
    mock_policy_list = [
        Policy(
            id=PolicyType.PROFANITY.value,
            name="Profanity Mock",
            state=True,
            action=Action.OVERRIDE.value,
            threshold=0.8,
            is_user_policy=True,
            is_llm_policy=True,
            message="Mock Profanity",
            pii_entities=None,
            pii_threshold=0.5,
            pii_categories=None,
            protected_prompts=None,
            prompt_leakage_threshold=0.85,
            locations=None,
            persons=None,
            competitors=None,
            metadata={},
        ),
        Policy(
            id=PolicyType.PII_LEAKAGE.value,
            name="PII Mock",
            state=True,
            action=Action.RETRY.value,
            pii_entities=["EMAIL_ADDRESS"],
            is_user_policy=True,
            is_llm_policy=True,
            message="Mock PII",
            pii_threshold=0.5,
            pii_categories=None,
            threshold=None,
            protected_prompts=None,
            prompt_leakage_threshold=0.85,
            locations=None,
            persons=None,
            competitors=None,
            metadata={},
        ),
    ]
    original_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[get_loaded_policies] = lambda: mock_policy_list
    yield mock_policy_list
    app.dependency_overrides = original_overrides


@pytest.fixture(scope="function")
def mock_app_state_for_proxy():
    """Mocks app_state components potentially used if validation runs."""
    mock_state = MagicMock()
    mock_state.presidio_analyzer_engine = MagicMock()
    mock_state.presidio_anonymizer_engine = MagicMock()
    mock_state.profanity_model = AsyncMock(spec=True)
    mock_state.ner_model = AsyncMock(spec=True)

    mock_state.profanity_model.reset_mock(return_value=True, side_effect=True)
    mock_state.ner_model.reset_mock(return_value=True, side_effect=True)
    mock_state.presidio_analyzer_engine.reset_mock(return_value=True, side_effect=True)

    if hasattr(mock_state.profanity_model, "predict"):
        mock_state.profanity_model.predict.return_value = ((0.9, 0.1), 5)
    if hasattr(mock_state.ner_model, "predict"):
        mock_state.ner_model.predict.return_value = ([], 5)
    if hasattr(mock_state.presidio_analyzer_engine, "analyze"):
        mock_state.presidio_analyzer_engine.analyze.return_value = []

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
        yield mock_state
    finally:
        for p in patchers:
            try:
                p.stop()
            except RuntimeError:
                pass


@pytest.fixture
def sample_policy_pii():
    """Return a sample PII policy suitable for OS Core (Observe mode)."""
    return Policy(
        id=PolicyType.PII_LEAKAGE.value,
        name="PII Detection",
        is_user_policy=True,
        is_llm_policy=True,
        action=Action.OBSERVE.value,
        message="Potential PII detected.",
        state=True,
        pii_entities=None,
        pii_threshold=0.5,
        pii_categories=None,
        threshold=None,
        protected_prompts=None,
        locations=None,
        persons=None,
        competitors=None,
        prompt_leakage_threshold=0.85,
        metadata={},
    )


@pytest.fixture
def sample_policy_prompt_leakage():
    """Return a sample prompt leakage policy for testing."""
    return Policy(
        id=PolicyType.PROMPT_LEAKAGE.value,
        name="Prompt Leakage Detection",
        is_user_policy=False,
        is_llm_policy=True,
        action=Action.OVERRIDE.value,
        message="Prompt leakage detected",
        state=True,
        protected_prompts=["secret system prompt", "confidential instructions"],
        prompt_leakage_threshold=0.9,
        threshold=None,
        pii_entities=None,
        pii_threshold=0.5,
        pii_categories=None,
        locations=None,
        persons=None,
        competitors=None,
        metadata={},
    )


@pytest.fixture
def sample_policy_profanity():
    """Return a sample profanity policy for testing."""
    return Policy(
        id=PolicyType.PROFANITY.value,
        name="Profanity Detection",
        is_user_policy=True,
        is_llm_policy=True,
        action=Action.OVERRIDE.value,
        message="Profanity detected",
        state=True,
        threshold=0.7,
        pii_entities=None,
        pii_threshold=0.5,
        pii_categories=None,
        protected_prompts=None,
        locations=None,
        persons=None,
        competitors=None,
        prompt_leakage_threshold=0.85,
        metadata={},
    )


@pytest.fixture
def sample_policy_competitor():
    """Return a sample competitor check policy for testing."""
    return Policy(
        id=PolicyType.COMPETITOR_CHECK.value,
        name="Competitor Check",
        is_user_policy=True,
        is_llm_policy=True,
        action=Action.OBSERVE.value,
        message="Competitor mentioned",
        state=True,
        competitors=["EvilCorp", "MegaGlobal"],
        threshold=0.8,
        pii_entities=None,
        pii_threshold=0.5,
        pii_categories=None,
        protected_prompts=None,
        locations=None,
        persons=None,
        prompt_leakage_threshold=0.85,
        metadata={},
    )


@pytest.fixture
def sample_policy_person():
    """Return a sample person check policy for testing."""
    return Policy(
        id=PolicyType.PERSON_CHECK.value,
        name="Person Check",
        is_user_policy=True,
        is_llm_policy=True,
        action=Action.OVERRIDE.value,
        message="Person mentioned",
        state=True,
        persons=["John Doe", "Jane Smith"],
        threshold=0.85,
        pii_entities=None,
        pii_threshold=0.5,
        pii_categories=None,
        protected_prompts=None,
        locations=None,
        competitors=None,
        prompt_leakage_threshold=0.85,
        metadata={},
    )


@pytest.fixture
def sample_policy_location():
    """Return a sample location check policy for testing."""
    return Policy(
        id=PolicyType.LOCATION_CHECK.value,
        name="Location Check",
        is_user_policy=True,
        is_llm_policy=True,
        action=Action.OBSERVE.value,
        message="Location mentioned",
        state=True,
        locations=["Paris", "Tokyo"],
        threshold=0.75,
        pii_entities=None,
        pii_threshold=0.5,
        pii_categories=None,
        protected_prompts=None,
        persons=None,
        competitors=None,
        prompt_leakage_threshold=0.85,
        metadata={},
    )


@pytest.fixture
def sample_messages():
    """Return a list of sample messages for testing."""
    return [
        {"role": "user", "content": "Hello", "user_id": "user-123"},
        {"role": "assistant", "content": "Hi there"},
    ]


@pytest.fixture
def sample_messages_with_pii():
    """Return sample messages containing potential PII patterns."""
    return [
        {
            "role": "user",
            "content": "My email is test@example.com",
            "user_id": "user-pii",
        },
        {"role": "assistant", "content": "Use card 4111-2222-3333-4444 please."},
        {"role": "user", "content": "Call 555-123-4567 maybe?"},
    ]


@pytest.fixture
def sample_message_long(tmp_path):
    """A long message likely requiring chunking."""
    max_chars = 1800
    overlap = 200
    target_len = max_chars + overlap
    base = "This is sentence number {i}. Some filler text repeated. "
    num_repeats = (target_len // len(base.format(i=1))) + 2
    long_content = "".join([base.format(i=i) for i in range(1, num_repeats)])
    pii_part = "My phone number is 987-654-3210 and email hidden@place.net."
    insert_pos = len(long_content) // 3
    final_content = long_content[:insert_pos] + pii_part + long_content[insert_pos:]
    return [{"role": "user", "content": final_content}]


@pytest.fixture
def sample_messages_with_prompt_leakage():
    """Return sample messages containing prompt leakage for testing."""
    return [
        {"role": "user", "content": "What does 'secret system prompt' mean?"},
        {
            "role": "assistant",
            "content": "The secret system prompt is confidential.",
        },
    ]


@pytest.fixture
def sample_messages_with_profanity():
    """Return sample messages containing profanity for testing."""
    return [
        {
            "role": "user",
            "content": "This is some damn offensive language!",
        },
        {"role": "assistant", "content": "Okay, understood."},
    ]


@pytest.fixture
def sample_messages_with_ner_entities():
    """Return sample messages containing entities for NER policies."""
    return [
        {
            "role": "user",
            "content": "Did John Doe from EvilCorp visit Paris last week?",
        },
        {
            "role": "assistant",
            "content": "I cannot confirm travel plans for Jane Smith to Tokyo.",
        },
    ]


@pytest.fixture
def temp_policies_file(tmp_path):
    """Creates a temporary policies.yaml file for testing policy loading."""
    default_policies_data = {
        "policies": [
            {
                "id": PolicyType.PII_LEAKAGE.value,
                "name": "Default PII",
                "state": True,
                "action": Action.OBSERVE.value,
                "message": "Default PII Message",
                "is_user_policy": True,
                "is_llm_policy": True,
            },
            {
                "id": PolicyType.PROFANITY.value,
                "name": "Default Profanity",
                "state": True,
                "action": Action.OVERRIDE.value,
                "threshold": 0.75,
                "message": "Default Profanity Message",
                "is_user_policy": True,
                "is_llm_policy": True,
            },
            {
                "id": PolicyType.PROMPT_LEAKAGE.value,
                "name": "Default Prompt Leakage",
                "state": True,
                "action": Action.OVERRIDE.value,
                "protected_prompts": ["secret internal code"],
                "prompt_leakage_threshold": 0.85,
                "message": "Default Prompt Leakage Message",
                "is_user_policy": False,
                "is_llm_policy": True,
            },
            {
                "id": PolicyType.COMPETITOR_CHECK.value,
                "name": "Default Competitor Check",
                "state": True,
                "action": Action.OBSERVE.value,
                "competitors": ["DefaultComp", "Another Inc"],
                "threshold": 0.8,
                "message": "Default Competitor Message",
                "is_user_policy": True,
                "is_llm_policy": True,
            },
            {
                "id": PolicyType.PERSON_CHECK.value,
                "name": "Default Person Check (Disabled)",
                "state": False,
                "action": Action.OVERRIDE.value,
                "persons": ["Known Person"],
                "threshold": 0.9,
                "message": "Default Person Message",
                "is_user_policy": True,
                "is_llm_policy": True,
            },
        ]
    }

    temp_file = tmp_path / "policies.yaml"
    with open(temp_file, "w", encoding="utf-8") as f:
        yaml.dump(default_policies_data, f)

    yield str(temp_file)
