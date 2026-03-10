from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional

from fastapi import status
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


class Action(IntEnum):
    OVERRIDE = 0
    OBSERVE = 1
    REDACT = 2
    RETRY = 3


class Policy(BaseModel):
    """Policy configuration loaded locally."""

    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
    )
    id: int
    name: str
    is_user_policy: bool = True
    is_llm_policy: bool = True
    action: int = Action.OVERRIDE.value
    message: str = "Policy violated."
    state: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

    threshold: Optional[float] = None

    locations: Optional[List[str]] = None
    persons: Optional[List[str]] = None
    competitors: Optional[List[str]] = None

    protected_prompts: Optional[List[str]] = None

    pii_categories: Optional[List[str]] = Field(
        default=None,
        description="Presidio categories (e.g., ['DEFAULT', 'US_SPECIFIC'])",
    )
    pii_entities: Optional[List[str]] = Field(
        default=None,
        description="Specific Presidio entity types (e.g., ['PERSON', 'US_SSN'])",
    )
    pii_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for PII detection",
    )
    prompt_leakage_threshold: Optional[float] = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for prompt leakage detection using fuzzy matching (0.0 to 1.0)",
    )

    @field_validator("action")
    @classmethod
    def check_action(cls, v: int) -> int:
        try:
            Action(v)
        except ValueError:
            raise ValueError(f"Invalid action value: {v}")
        return v

    @field_validator("id")
    @classmethod
    def check_policy_type(cls, v: int) -> int:
        try:
            PolicyType(v)
        except ValueError:
            raise ValueError(f"Invalid policy id (type): {v}")
        return v


@dataclass
class Agent:
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class PolicyType(IntEnum):
    PII_LEAKAGE = 1
    PROMPT_LEAKAGE = 2
    COMPETITOR_CHECK = 3
    PERSON_CHECK = 4
    LOCATION_CHECK = 5
    PROFANITY = 6


@dataclass
class SafetyCode:
    TIMEOUT = -80
    UNEXPECTED = -70
    GENERIC_UNSAFE = -10
    SAFE = 0
    PII_DETECTED = 10
    PROMPT_LEAKED = 20
    COMPETITOR_DETECTED = 30
    PERSON_DETECTED = 40
    LOCATION_DETECTED = 50
    PROFANE = 60


@dataclass
class Status:
    status: int
    message: str
    safety_code: int = SafetyCode.SAFE
    action: Optional[int] = None
    processed_content: Optional[str] = None


class Result:
    @staticmethod
    def safe_result() -> Status:
        return Status(
            status=status.HTTP_200_OK,
            message="Message validated successfully",
            safety_code=SafetyCode.SAFE,
            action=None,
            processed_content=None,
        )

    @staticmethod
    def unsafe_result(
        message: str,
        safety_code: int,
        action: int,
        status_code: int = status.HTTP_200_OK,
        processed_content: Optional[str] = None,
    ) -> Status:
        try:
            Action(action)
        except ValueError:
            action = Action.OVERRIDE.value
        return Status(
            message=message,
            safety_code=safety_code,
            status=status_code,
            action=action,
            processed_content=processed_content,
        )
