import logging

from aiolimiter import AsyncLimiter

from assistant.memory.graphrag_v1.llm.types import LLMConfig
from assistant.memory.graphrag_v1.llm.limiting.llm_limiter import LLMLimiter
from assistant.memory.graphrag_v1.llm.limiting.tpm_rpm_limiter import TpmRpmLLMLimiter

log = logging.getLogger(__name__)


def create_tpm_rpm_limiters(
        configuration: LLMConfig
) -> LLMLimiter:
    """
    创建模型速率限制器
    :param configuration: 模型配置
    :return: 模型速率限制器
    """
    tpm = configuration.tokens_per_minute
    rpm = configuration.requests_per_minute

    return TpmRpmLLMLimiter(
        None if tpm == 0 else AsyncLimiter(tpm or 50_000),
        None if rpm == 0 else AsyncLimiter(rpm or 10_000),
    )
