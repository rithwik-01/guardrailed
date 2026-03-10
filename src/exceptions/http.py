from fastapi import HTTPException

from src.shared import Action, SafetyCode


class GuardrailedHTTPException(HTTPException):
    def __init__(
        self,
        status_code: int,
        message: str = "An error occurred.",
        safety_code: int = SafetyCode.UNEXPECTED,
        action: int = Action.OVERRIDE,
    ):
        super().__init__(status_code=status_code, detail=message)
        self.safety_code = safety_code
        self.message = message
        self.action = action
