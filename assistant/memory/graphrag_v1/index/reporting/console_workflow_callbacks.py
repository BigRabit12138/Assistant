from datashaper import NoopWorkflowCallbacks


class ConsoleWorkflowCallbacks(NoopWorkflowCallbacks):
    def on_error(
        self,
        message: str,
        cause: BaseException | None = None,
        stack: str | None = None,
        details: dict | None = None,
    ) -> None:
        print(message, str(cause), stack, details)

    def on_warning(
            self,
            message: str,
            details: dict | None = None
    ) -> None:
        _print_warning(message)

    def on_log(
            self,
            message: str,
            details: dict | None = None
    ) -> None:
        print(message, details)


def _print_warning(skk):
    print(f"\033[93m {skk}\033[00m")
