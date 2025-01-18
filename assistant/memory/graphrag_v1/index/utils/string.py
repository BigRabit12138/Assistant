import re
import html

from typing import Any


def clean_str(input_: Any) -> str:
    """
    过滤文本
    :param input_: 输入文本
    :return: 过滤的文本
    """
    if not isinstance(input_, str):
        return input_

    result = html.unescape(input_.strip())

    return re.sub(r'[\x00-\x1f\x7f-\x9f]', "", result)
