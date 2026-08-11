from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.shared import Policy


@dataclass
class ValidationContext:
    policies: List[Policy]
    messages: List[Dict[str, str]]
    user_id: Optional[str] = None

    # Populated by the validator when a policy with Action.REDACT fires:
    # message index -> redacted text to forward in place of the original.
    redactions: Dict[int, str] = field(default_factory=dict)
