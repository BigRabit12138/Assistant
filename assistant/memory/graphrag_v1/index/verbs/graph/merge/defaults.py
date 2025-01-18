from assistant.memory.graphrag_v1.index.verbs.graph.merge.typing import BasicMergeOperation


DEFAULT_NODE_OPERATIONS = {
    "*": {
        "operation": BasicMergeOperation.Replace,
    }
}

DEFAULT_EDGE_OPERATIONS = {
    "*": {
        "operation": BasicMergeOperation.Replace
    },
    "weight": "sum",
}

DEFAULT_CONCAT_SEPARATOR = ","
