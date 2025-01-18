class NoWorkflowsDefinedError(ValueError):
    def __init__(self):
        super().__init__("No workflows defined.")


class UndefinedWorkflowError(ValueError):
    def __init__(self):
        super().__init__("Workflow name is undefined.")


class UnknownWorkflowError(ValueError):
    def __init__(self, name: str):
        super().__init__(f"Unknown workflow: {name}")
