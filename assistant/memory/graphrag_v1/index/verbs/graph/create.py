from typing import Any

import pandas as pd
import networkx as nx

from datashaper import (
    verb,
    VerbInput,
    VerbCallbacks,
    TableContainer,
    progress_iterable,
)

from assistant.memory.graphrag_v1.index.utils import clean_str

DEFAULT_NODE_ATTRIBUTES = ["label", "type", "id", "name", "description", "community"]
DEFAULT_EDGE_ATTRIBUTES = ["label", "type", "name", "source", "target"]


@verb(name="create_graph")
def create_graph(
        input: VerbInput,
        callbacks: VerbCallbacks,
        to: str,
        type_: str,
        graph_type: str = "undirected",
        **kwargs,
) -> TableContainer:
    if type_ != "node" and type_ != "edge":
        msg = f"Unknown type {type_}."
        raise ValueError(msg)
    input_df = input.get_input()
    num_total = len(input_df)
    out_graph: nx.Graph = _create_nx_graph(graph_type)

    in_attributes = (
        _get_node_attributes(kwargs) if type_ == "node" else _get_edge_attributes(kwargs)
    )

    id_col = in_attributes.get(
        "id", in_attributes.get("label", in_attributes.get("name", None))
    )
    source_col = in_attributes.get("source", None)
    target_col = in_attributes.get("target", None)

    for _, row in progress_iterable(input_df.iterrows(), callbacks.progress, num_total):
        item_attributes = {
            clean_str(key): _clean_value(row[value])
            for key, value in in_attributes.items()
            if value in row
        }
        if type_ == "node":
            id_ = clean_str(row[id_col])
            out_graph.add_node(id_, **item_attributes)
        elif type_ == "edge":
            source = clean_str(row[source_col])
            target = clean_str(row[target_col])
            out_graph.add_edge(source, target, **item_attributes)

        graphml_string = "".join(nx.generate_graphml(out_graph))
        output_df = pd.DataFrame([{to: graphml_string}])
        return TableContainer(table=output_df)


def _clean_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return clean_str(value)

    msg = f"Value must be a string or None, got {type(value)}."
    raise TypeError(msg)


def _get_node_attributes(args: dict[str, Any]) -> dict[str, Any]:
    mapping = _get_attribute_column_mapping(
        args.get("attributes", DEFAULT_NODE_ATTRIBUTES)
    )
    if "id" not in mapping and "label" not in mapping and "name" not in mapping:
        msg = "You must specify an id, label, or name column in the node attributes."
        raise ValueError(msg)
    return mapping


def _get_edge_attributes(args: dict[str, Any]) -> dict[str, Any]:
    mapping = _get_attribute_column_mapping(
        args.get("attributes", DEFAULT_EDGE_ATTRIBUTES)
    )
    if "source" not in mapping or "target" not in mapping:
        msg = "You must specify a source and target column in the edge attributes."
        raise ValueError(msg)
    return mapping


def _get_attribute_column_mapping(
        in_attributes: dict[str, Any] | list[str],
) -> dict[str, str]:
    if isinstance(in_attributes, dict):
        return {
            **in_attributes,
        }

    return {attrib: attrib for attrib in in_attributes}


def _create_nx_graph(graph_type: str) -> nx.Graph:
    if graph_type == "directed":
        return nx.DiGraph()

    return nx.Graph()
