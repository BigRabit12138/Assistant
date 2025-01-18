from __future__ import annotations

from abc import ABC, abstractmethod

from datashaper import Progress


class ProgressReporter(ABC):
    @abstractmethod
    def __call__(self, update: Progress):
        pass

    @abstractmethod
    def dispose(self):
        pass

    @abstractmethod
    def child(
            self,
            prefix: str,
            transient=True
    ) -> ProgressReporter:
        pass

    @abstractmethod
    def force_refresh(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def error(self, message: str) -> None:
        pass

    @abstractmethod
    def warning(self, message: str) -> None:
        pass

    @abstractmethod
    def info(self, message: str) -> None:
        pass

    @abstractmethod
    def success(self, message: str) -> None:
        pass


class NullProgressReporter(ProgressReporter):
    def __call__(self, update: Progress) -> None:
        pass

    def dispose(self) -> None:
        pass

    def child(
            self,
            predix: str,
            transient: bool = True
    ) -> ProgressReporter:
        return self

    def force_refresh(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def error(self, message: str) -> None:
        pass

    def warning(self, message: str) -> None:
        pass

    def info(self, message: str) -> None:
        pass

    def success(self, message: str) -> None:
        pass


class PrintProgressReporter(ProgressReporter):
    prefix: str

    def __init__(self, prefix: str):
        self.prefix = prefix
        print(f"\n{self.prefix}", end="")

    def __call__(self, update: Progress) -> None:
        print(".", end="")

    def dispose(self) -> None:
        pass

    def child(
            self,
            prefix: str,
            transient=True
    ) -> ProgressReporter:
        return PrintProgressReporter(prefix)

    def stop(self) -> None:
        pass

    def force_refresh(self) -> None:
        pass

    def error(self, message: str) -> None:
        print(f"\n{self.prefix}ERROR: {message}")

    def warning(self, message: str) -> None:
        print(f"\n{self.prefix}WARNING: {message}")

    def info(self, message: str) -> None:
        print(f"\n{self.prefix}INFO: {message}")

    def success(self, message: str) -> None:
        print(f"\n{self.prefix}SUCCESS: {message}")

