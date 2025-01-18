from collections.abc import Callable

from assistant.memory.graphrag_v1.llm.types.llm_invocation_result import LLMInvocationResult

ErrorHandlerFn = Callable[[BaseException | None, str | None, dict | None], None]

LLMInvocationFn = Callable[[LLMInvocationResult], None]

OnCacheActionFn = Callable[[str, str | None], None]

IsResponseValidFn = Callable[[dict], bool]
