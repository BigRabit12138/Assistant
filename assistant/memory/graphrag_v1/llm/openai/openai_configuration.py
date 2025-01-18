import json

from typing import Any, cast
from collections.abc import Hashable

from assistant.memory.graphrag_v1.llm.types import LLMConfig


def _non_blank(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()

    return None if stripped == "" else value


class OpenAIConfiguration(Hashable, LLMConfig):
    """
    openai大模型配置
    """
    _api_key: str
    _model: str

    _api_base: str | None
    _api_version: str | None
    _cognitive_services_endpoint: str | None
    _deployment_name: str | None
    _organization: str | None
    _proxy: str | None

    _n: int | None
    _temperature: float | None
    _frequency_penalty: float | None
    _presence_penalty: float | None
    _top_p: float | None
    _max_tokens: int | None
    _response_format: str | None
    _logit_bias: dict[str, float] | None
    _stop: list[str] | None

    _max_retries: int | None
    _max_retry_wait: float | None
    _request_timeout: float | None

    _raw_config: dict

    _model_supports_json: bool | None

    _tokens_per_minute: int | None
    _request_per_minute: int | None
    _concurrent_requests: int | None
    _encoding_model: str | None
    _sleep_on_rate_limit_recommendation: bool | None

    def __init__(
            self,
            config: dict,
    ):
        def lookup_required(key: str) -> str:
            return cast(str, config.get(key))

        def lookup_str(key: str) -> str | None:
            return cast(str | None, config.get(key))

        def lookup_int(key: str) -> int | None:
            result = config.get(key)
            if result is None:
                return None

            return int(cast(int, result))

        def lookup_float(key: str) -> float | None:
            result = config.get(key)
            if result is None:
                return None
            return float(cast(float, result))

        def lookup_dict(key: str) -> dict | None:
            return cast(dict | None, config.get(key))

        def lookup_list(key: str) -> list | None:
            return cast(list | None, config.get(key))

        def lookup_bool(key: str) -> bool | None:
            value = config.get(key)
            if isinstance(value, str):
                return value.upper() == "TRUE"
            if isinstance(value, int):
                return value > 0

            return cast(bool | None, config.get(key))

        self._api_key = lookup_required("api_key")
        self._model = lookup_required("model")
        self._deployment_name = lookup_str("deployment_name")
        self._api_base = lookup_str("api_base")
        self._api_version = lookup_str("api_version")
        self._cognitive_services_endpoint = lookup_str("cognitive_services_endpoint")
        self._organization = lookup_str("organization")
        self._proxy = lookup_str("proxy")
        self._n = lookup_int("n")
        self._temperature = lookup_float("temperature")
        self._frequency_penalty = lookup_float("frequency_penalty")
        self._presence_penalty = lookup_float("presence_penalty")
        self._top_p = lookup_float("top_p")
        self._max_tokens = lookup_int("max_tokens")
        self._response_format = lookup_str("response_format")
        self._logit_bias = lookup_dict("logit_bias")
        self._stop = lookup_list("stop")
        self._max_retries = lookup_int("max_retries")
        self._request_timeout = lookup_float("request_timeout")
        self._model_supports_json = lookup_bool("model_supports_json")
        self._tokens_per_minute = lookup_int("tokens_per_minute")
        self._requests_per_minute = lookup_int("requests_per_minute")
        self._concurrent_requests = lookup_int("concurrent_requests")
        self._encoding_model = lookup_str("encoding_model")
        self._max_retry_wait = lookup_float("max_retry_wait")
        self._sleep_on_rate_limit_recommendation = lookup_bool(
            "sleep_on_rate_limit_recommendation"
        )
        self._raw_config = config

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def model(self) -> str:
        return self._model

    @property
    def deployment_name(self) -> str | None:
        return _non_blank(self._deployment_name)

    @property
    def api_base(self) -> str | None:
        result = _non_blank(self._api_base)
        return result[:-1] if result and result.endswith("/") else result

    @property
    def api_version(self) -> str | None:
        return _non_blank(self._api_version)

    @property
    def cognitive_service_endpoint(self) -> str | None:
        return _non_blank(self._cognitive_services_endpoint)

    @property
    def organization(self) -> str | None:
        return _non_blank(self._organization)

    @property
    def proxy(self) -> str | None:
        return _non_blank(self._proxy)

    @property
    def n(self) -> int | None:
        return self._n

    @property
    def temperature(self) -> float | None:
        return self._temperature

    @property
    def frequency_penalty(self) -> float | None:
        return self._frequency_penalty

    @property
    def presence_penalty(self) -> float | None:
        return self._presence_penalty

    @property
    def top_p(self) -> float | None:
        return self._top_p

    @property
    def max_tokens(self) -> int | None:
        return self._max_tokens

    @property
    def response_format(self) -> str | None:
        return _non_blank(self._response_format)

    @property
    def logit_bias(self) -> dict[str, float] | None:
        return self._logit_bias

    @property
    def stop(self) -> list[str] | None:
        return self._stop

    @property
    def max_retries(self) -> int | None:
        return self._max_retries

    @property
    def max_retry_wait(self) -> float | None:
        return self._max_retry_wait

    @property
    def request_timeout(self) -> float | None:
        return self._request_timeout

    @property
    def model_supports_json(self) -> bool | None:
        return self._model_supports_json

    @property
    def tokens_per_minute(self) -> int | None:
        return self._tokens_per_minute

    @property
    def requests_per_minute(self) -> int | None:
        return self._requests_per_minute

    @property
    def concurrent_requests(self) -> int | None:
        return self._concurrent_requests

    @property
    def encoding_model(self) -> str | None:
        return _non_blank(self._encoding_model)

    @property
    def sleep_on_rate_limit_recommendation(self) -> bool | None:
        return self._sleep_on_rate_limit_recommendation

    @property
    def raw_config(self) -> dict:
        return self._raw_config

    def lookup(
            self,
            name: str,
            default_value: Any = None
    ) -> Any:
        return self._raw_config.get(name, default_value)

    def __str__(self) -> str:
        return json.dumps(self.raw_config, indent=4)

    def __repr__(self) -> str:
        return f"OpenAIConfiguration({self._raw_config})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OpenAIConfiguration):
            return False

        return self._raw_config == other._raw_config

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._raw_config.items())))
