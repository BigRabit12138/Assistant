from typing import Any
from hashlib import md5
from collections.abc import Iterable


def gen_md5_hash(
        item: dict[str, Any],
        hashcode: Iterable[str]
):
    """
    选择指定的元素的内容，计算MD5
    :param item: 全部内容
    :param hashcode: 指定的元素
    :return: MD5 hash值
    """
    hashed = "".join([str(item[column]) for column in hashcode])
    return f"{md5(hashed.encode('utf-8'), usedforsecurity=False).hexdigest()}"
