class RetriesExhaustedError(RuntimeError):
    def __init__(
            self,
            name: str,
            num_retries: int
    ) -> None:
        super().__init__(f"Operation '{name}' failed - {num_retries} retries exhausted.")
