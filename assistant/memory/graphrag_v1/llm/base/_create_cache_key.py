import hashlib


def _llm_string(params: dict) -> str:
    """
    格式化模型参数
    :param params: 模型参数
    :return: 参数字符串
    """
    if "max_tokens" in params and "n" not in params:
        params["n"] = None
    return str(sorted((k, v) for k, v in params.items()))


def _hash(_input: str) -> str:
    """
    计算MD5 Hash
    :param _input: 文本
    :return: MD5 Hash
    """
    return hashlib.md5(_input.encode()).hexdigest()


def create_hash_key(
        operation: str,
        prompt: str,
        parameters: dict
) -> str:
    """
    创建输入内容Hash ID
    :param operation: 图操作类型
    :param prompt: 模板
    :param parameters: 模型参数
    :return: 缓存标识ID
    """
    llm_string = _llm_string(parameters)
    return f"{operation}--{_hash(prompt + llm_string)}"
