from enum import Enum
from dataclasses import dataclass


class BasicMergeOperation(str, Enum):
    """
    基础合并方法
    """
    Replace = "replace"
    Skip = "skip"


class StringOperation(str, Enum):
    """
    字符串合并方法
    """
    Concat = "concat"
    Replace = "replace"
    Skip = "skip"


class NumericOperation(str, Enum):
    """
    数字合并方法
    """
    Sum = "sum"
    Average = "average"
    Max = "max"
    Min = "min"
    Multiply = "multiply"
    Replace = "replace"
    Skip = "skip"


@dataclass
class DetailedAttributeMergeOperation:
    """
    具体的属性合并操作数据结构
    """
    operation: str

    separator: str | None = None
    delimiter: str | None = None
    distinct: bool = False


# 属性合并操作数据结构
AttributeMergeOperation = str | DetailedAttributeMergeOperation
