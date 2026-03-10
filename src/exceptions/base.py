class GuardrailedBaseError(Exception):
    def __init__(self, message: str, status_code: int, *, user_facing: bool = False):
        self.message = message
        self.status_code = status_code
        self.user_facing = user_facing
        super().__init__(self.message)
