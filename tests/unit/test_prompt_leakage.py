"""
Tests for prompt leakage detection functionality.
"""

from src.domain.validators.prompt_leakage.main import check_prompt
from src.shared import Policy, PolicyType, SafetyCode


def test_check_prompt_no_leakage():
    """
    Test prompt leakage detection with no protected prompts.
    """
    message = "This is a normal message with no prompt leakage."
    policy = Policy(
        id=PolicyType.PROMPT_LEAKAGE,
        name="Prompt Leakage Detection",
        is_user_policy=False,
        is_llm_policy=True,
        action=0,
        message="Prompt leakage detected.",
        state=True,
        protected_prompts=[],
    )

    result = check_prompt(message, policy)

    assert result.safety_code == SafetyCode.SAFE
    assert result.status == 200
    assert result.message == "Message validated successfully"


def test_check_prompt_with_protected_prompt():
    """
    Test prompt leakage detection with a matching protected prompt.
    """
    protected_prompt = "this is a secret system prompt"
    message = f"I found that {protected_prompt} in the documentation."

    policy = Policy(
        id=PolicyType.PROMPT_LEAKAGE,
        name="Prompt Leakage Detection",
        is_user_policy=False,
        is_llm_policy=True,
        action=0,
        message="Prompt leakage detected.",
        state=True,
        protected_prompts=[protected_prompt],
    )

    result = check_prompt(message, policy)

    assert result.safety_code == SafetyCode.PROMPT_LEAKED
    assert result.status == 200
    assert result.message == "Prompt leakage detected."


def test_check_prompt_case_insensitive():
    """
    Test that prompt leakage detection is case insensitive.
    """
    protected_prompt = "SECRET SYSTEM PROMPT"
    message = "I found that secret system prompt in the documentation."

    policy = Policy(
        id=PolicyType.PROMPT_LEAKAGE,
        name="Prompt Leakage Detection",
        is_user_policy=False,
        is_llm_policy=True,
        action=0,
        message="Prompt leakage detected.",
        state=True,
        protected_prompts=[protected_prompt],
    )

    result = check_prompt(message, policy)

    assert result.safety_code == SafetyCode.PROMPT_LEAKED
    assert result.status == 200
    assert result.message == "Prompt leakage detected."


def test_check_prompt_multiple_protected():
    """
    Test prompt leakage detection with multiple protected prompts.
    """
    protected_prompts = [
        "secret system prompt A",
        "secret system prompt B",
        "secret system prompt C",
    ]

    message = "This message contains secret system prompt B which should be detected."

    policy = Policy(
        id=PolicyType.PROMPT_LEAKAGE,
        name="Prompt Leakage Detection",
        is_user_policy=False,
        is_llm_policy=True,
        action=0,
        message="Prompt leakage detected.",
        state=True,
        protected_prompts=protected_prompts,
    )

    result = check_prompt(message, policy)

    assert result.safety_code == SafetyCode.PROMPT_LEAKED
    assert result.status == 200
    assert result.message == "Prompt leakage detected."

    safe_message = "This message is safe and contains no protected prompts."
    safe_result = check_prompt(safe_message, policy)

    assert safe_result.safety_code == SafetyCode.SAFE
    assert safe_result.status == 200
    assert safe_result.message == "Message validated successfully"


def test_check_prompt_custom_message():
    """
    Test prompt leakage detection with a custom message.
    """
    protected_prompt = "secret system prompt"
    message = f"I found that {protected_prompt} in the documentation."
    custom_message = "Custom prompt leakage message."

    policy = Policy(
        id=PolicyType.PROMPT_LEAKAGE,
        name="Prompt Leakage Detection",
        is_user_policy=False,
        is_llm_policy=True,
        action=0,
        message=custom_message,
        state=True,
        protected_prompts=[protected_prompt],
    )

    result = check_prompt(message, policy)

    assert result.safety_code == SafetyCode.PROMPT_LEAKED
    assert result.status == 200
    assert result.message == custom_message
