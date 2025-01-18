from typing import cast

from datashaper import TableContainer, VerbInput

_NAMED_INPUTS_REQUIRED = "Named inputs are required"


def get_required_input_table(
        input_: VerbInput,
        name: str
) -> TableContainer:
    """
    从动作输入中获取指定名字的表格
    :param input_: 输入，包含表格
    :param name: 指定表的名字
    :return: 指定表格
    """
    return cast(TableContainer, get_named_input_table(
        input_,
        name,
        required=True
    ))


def get_named_input_table(
        input_: VerbInput,
        name: str,
        required: bool = False
) -> TableContainer | None:
    """
    从动作输入中获取指定名字的表格
    :param input_: 输入，包含表格
    :param name: 指定表的名字
    :param required: 是否必须
    :return: 指定表格
    """
    named_inputs = input_.named
    if named_inputs is None:
        if not required:
            return None
        raise ValueError(_NAMED_INPUTS_REQUIRED)

    result = named_inputs.get(name)
    if result is None and required:
        msg = f"input '${name}' is required."
        raise ValueError(msg)

    return result
