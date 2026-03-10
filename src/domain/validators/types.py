import logging
from dataclasses import dataclass, field
from typing import List, Optional

from src.shared import PolicyType

logger = logging.getLogger(__name__)


@dataclass
class LocalPolicyResult:
    policy_type: PolicyType
    is_violated: bool


@dataclass
class ContentMessage:
    """Represents a message being validated with its metadata"""

    content: str
    user_id: Optional[str] = None
    token_count: int = 0
    policy_violations: List[PolicyType] = field(default_factory=list)

    def add_violation(self, policy_type: PolicyType) -> None:
        """Record a policy violation for this message."""
        if policy_type not in self.policy_violations:
            self.policy_violations.append(policy_type)
            logger.debug(
                f"Violation recorded for policy type {policy_type.name} on message."
            )
