from enum import Enum
from typing import Any, TypeVar
from collections.abc import Callable
from contextlib import contextmanager

from environs import Env

T = TypeVar("T")

KeyValue = str | Enum
EnvKeySet = str | list[str]


def read_key(value: KeyValue) -> str:
    if not isinstance(value, str):
        return value.value.lower()

    return value.lower()


class EnvironmentReader:
    _env: Env
    _config_stack: list[dict]

    def __init__(self, env: Env):
        self._env = env
        self._config_stack = []

    @property
    def env(self):
        return self._env

    @staticmethod
    def _read_env(
            env_key: str | list[str],
            default_value: T,
            read: Callable[[str, T], T]
    ) -> T | None:
        if isinstance(env_key, str):
            env_key = [env_key]

        for k in env_key:
            result = read(k.upper(), default_value)
            if result is not default_value:
                return result

        return default_value

    def envvar_prefix(self, prefix: KeyValue):
        prefix = read_key(prefix)
        prefix = f"{prefix}_".upper()
        return self._env.prefixed(prefix)

    def use(self, value: Any | None):
        @contextmanager
        def config_context():
            self._config_stack.append(value or {})
            try:
                yield
            finally:
                self._config_stack.pop()

        return config_context()

    @property
    def section(self) -> dict:
        return self._config_stack[-1] if self._config_stack else {}

    def str(
            self,
            key: KeyValue,
            env_key: EnvKeySet | None = None,
            default_value: str | None = None
    ) -> str | None:
        key = read_key(key)
        if self.section and key in self.section:
            return self.section[key]

        return self._read_env(
            env_key or key,
            default_value,
            (lambda k, dv: self._env(k, dv))
        )

    def int(
            self,
            key: KeyValue,
            env_key: EnvKeySet | None = None,
            default_value: int | None = None,
    ) -> int | None:
        key = read_key(key)
        if self.section and key in self.section:
            return int(self.section[key])

        return self._read_env(
            env_key or key,
            default_value,
            lambda k, dv: self._env.int(k, dv)
        )

    def bool(
            self,
            key: KeyValue,
            env_key: EnvKeySet | None = None,
            default_value: bool | None = None,
    ) -> bool | None:
        key = read_key(key)
        if self.section and key in self.section:
            return bool(self.section[key])

        return self._read_env(
            env_key or key,
            default_value,
            lambda k, dv: self._env.bool(k, dv)
        )

    def float(
            self,
            key: KeyValue,
            env_key: EnvKeySet | None = None,
            default_value: float | None = None,
    ) -> float | None:
        key = read_key(key)
        if self.section and key in self.section:
            return float(self.section[key])

        return self._read_env(
            env_key or key,
            default_value,
            lambda k, dv: self._env.float(k, dv)
        )

    def list(
            self,
            key: KeyValue,
            env_key: EnvKeySet | None = None,
            default_value: list | None = None,
    ) -> list | None:
        key = read_key(key)
        result = None
        if self.section and key in self.section:
            result = self.section[key]
            if isinstance(result, list):
                return result

        if result is None:
            result = self.str(key, env_key)
        if result:
            return [s.strip() for s in result.split(',')]

        return default_value
