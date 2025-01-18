from typing import Any
from abc import ABCMeta, abstractmethod


class StatusReporter(metaclass=ABCMeta):
    @abstractmethod
    def error(
            self,
            message: str,
            details: dict[str, Any] | None = None
    ):
        pass

    @abstractmethod
    def warning(
            self,
            message: str,
            details: dict[str, Any] | None = None
    ):
        pass

    @abstractmethod
    def log(
            self,
            message: str,
            details: dict[str, Any] | None = None
    ):
        pass


class ConsoleStatusReporter(StatusReporter):
    def error(
            self,
            message: str,
            details: dict[str, Any] | None = None
    ):
        print(message, details)

    def warning(
            self,
            message: str,
            details: dict[str, Any] | None = None
    ):
        _print_warning(message)

    def log(
            self,
            message: str,
            details: dict[str, Any] | None = None
    ):
        print(message, details)


def _print_warning(skk):
    print(f"\033[93m {skk}\033[00m")

