from enum import Enum


class TableEmitterType(str, Enum):

    Json = "json"
    Parquet = "parquet"
    CSV = "csv"
