from dataclasses import dataclass
from typing import Dict, List, Optional

from src.shared import Policy


@dataclass
class ValidationContext:
    policies: List[Policy]
    messages: List[Dict[str, str]]
    user_id: Optional[str] = None
