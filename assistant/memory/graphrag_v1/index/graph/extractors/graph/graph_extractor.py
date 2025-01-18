import re
import logging
import numbers
import traceback

from typing import Any
from dataclasses import dataclass
from collections.abc import Mapping

import tiktoken

import networkx as nx

import assistant.memory.graphrag_v1.config.defaults as defaults

from assistant.memory.graphrag_v1.llm import CompletionLLM
from assistant.memory.graphrag_v1.index.utils import clean_str
from assistant.memory.graphrag_v1.index.typing import ErrorHandlerFn
from assistant.memory.graphrag_v1.index.graph.extractors.graph.prompts import (
    LOOP_PROMPT,
    CONTINUE_PROMPT,
    GRAPH_EXTRACTION_PROMPT,
)

DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"
DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]


@dataclass
class GraphExtractionResult:
    """
    图对象提取结果
    """
    output: nx.Graph
    source_docs: dict[Any, Any]


class GraphExtractor:
    """
    图对象提取器
    """
    _llm: CompletionLLM
    _join_descriptions: bool
    _tuple_delimiter_key: str
    _record_delimiter_key: str
    _entity_types_key: str
    _input_text_key: str
    _completion_delimiter_key: str
    _entity_name_key: str
    _input_description_key: str
    _extraction_prompt: str
    _summarization_prompt: str
    _loop_args: dict[str, Any]
    _max_gleanings: int
    _on_error: ErrorHandlerFn

    def __init__(
            self,
            llm_invoker: CompletionLLM,
            tuple_delimiter_key: str | None = None,
            record_delimiter_key: str | None = None,
            input_text_key: str | None = None,
            entity_types_key: str | None = None,
            completion_delimiter_key: str | None = None,
            prompt: str | None = None,
            join_descriptions=True,
            encoding_model: str | None = None,
            max_gleanings: int | None = None,
            on_error: ErrorHandlerFn | None = None,
    ):
        self._llm = llm_invoker
        self._join_descriptions = join_descriptions
        self._input_text_key = input_text_key or "input_text"
        self._tuple_delimiter_key = tuple_delimiter_key or "tuple_delimiter"
        self._record_delimiter_key = record_delimiter_key or "record_delimiter"
        self._completion_delimiter_key = (
            completion_delimiter_key or "completion_delimiter"
        )
        self._entity_types_key = entity_types_key or "entity_types"
        self._extraction_prompt = prompt or GRAPH_EXTRACTION_PROMPT
        self._max_gleanings = (
            max_gleanings
            if max_gleanings is not None
            else defaults.ENTITY_EXTRACTION_MAX_GLEANINGS
        )
        self._on_error = on_error or (lambda _e, _s, _d: None)

        encoding = tiktoken.get_encoding(encoding_model or "cl100k_base")
        yes = encoding.encode("YES")
        no = encoding.encode("NO")
        self._loop_args = {"logit_bias": {yes[0]: 100, no[0]: 100}, "max_tokens": 1}

    async def __call__(
            self,
            texts: list[str],
            prompt_variables: dict[str, Any] | None = None
    ) -> GraphExtractionResult:
        """
        提取图对象
        :param texts: 文本
        :param prompt_variables: prompt模板
        :return: 图提取结果
        """
        if prompt_variables is None:
            prompt_variables = {}
        all_records: dict[int, str] = {}
        source_doc_map: dict[int, str] = {}

        prompt_variables = {
            **prompt_variables,
            self._tuple_delimiter_key: prompt_variables.get(self._tuple_delimiter_key)
            or DEFAULT_TUPLE_DELIMITER,
            self._record_delimiter_key: prompt_variables.get(self._record_delimiter_key)
            or DEFAULT_RECORD_DELIMITER,
            self._completion_delimiter_key: prompt_variables.get(
                self._completion_delimiter_key
            )
            or DEFAULT_COMPLETION_DELIMITER,
            self._entity_types_key: ",".join(
                prompt_variables[self._entity_types_key] or DEFAULT_ENTITY_TYPES
            ),
        }

        for doc_index, text in enumerate(texts):
            try:
                # 提取图对象
                result = await self._process_document(text, prompt_variables)
                source_doc_map[doc_index] = text
                all_records[doc_index] = result
            except Exception as e:
                logging.exception("error extracting graph.")
                self._on_error(
                    e,
                    traceback.format_exc(),
                    {
                        "doc_index": doc_index,
                        "text": text,
                    },
                )

        # 解析图对象
        output = await self._process_results(
            all_records,
            prompt_variables.get(self._tuple_delimiter_key, DEFAULT_TUPLE_DELIMITER),
            prompt_variables.get(self._record_delimiter_key, DEFAULT_RECORD_DELIMITER),
        )

        return GraphExtractionResult(
            output=output,
            source_docs=source_doc_map,
        )

    async def _process_document(
            self,
            text: str,
            prompt_variables: dict[str, str]
    ) -> str:
        """
        对一块文本进行图对象提取
        :param text: 文本
        :param prompt_variables: prompt模板
        :return: 大模型输出
        """
        # 调用模型
        response = await self._llm(
            self._extraction_prompt,
            variables={
                **prompt_variables,
                self._input_text_key: text
            },
        )
        results = response.output or ""

        # 尽可能多的尝试提取
        for i in range(self._max_gleanings):
            glean_response = await self._llm(
                CONTINUE_PROMPT,
                name=f"extract-continuation-{i}",
                history=response.history or [],
            )
            results += glean_response.output or ""

            if i >= self._max_gleanings - 1:
                break

            # 判断是否继续
            continuation = await self._llm(
                LOOP_PROMPT,
                name=f"extract-loopcheck-{i}",
                history=glean_response.history or [],
                model_parameters=self._loop_args,
            )
            if continuation.output != "YES":
                break
        return results

    async def _process_results(
            self,
            results: dict[int, str],
            tuple_delimiter: str,
            record_delimiter: str,
    ) -> nx.Graph:
        """
        将模型输出的文本解析文一个图对象
        :param results: 模型输出文本
        :param tuple_delimiter: 内分割符
        :param record_delimiter: 外分割符
        :return: 图对象
        """
        graph = nx.Graph()
        for source_doc_id, extracted_data in results.items():
            # 分割为多个实体
            records = [r.strip() for r in extracted_data.split(record_delimiter)]

            for record in records:
                # 去掉括号
                record = re.sub(r"^\(|\)$", "", record.strip())
                # 分割为实体属性
                record_attributes = record.split(tuple_delimiter)

                # 图对象为实体
                if record_attributes[0] == '"entity"' and len(record_attributes) >= 4:
                    # 获取实体名称
                    entity_name = clean_str(record_attributes[1].upper())
                    # 获取实体类型
                    entity_type = clean_str(record_attributes[2].upper())
                    # 获取实体描述
                    entity_description = clean_str(record_attributes[3])

                    if entity_name in graph.nodes():
                        node = graph.nodes[entity_name]
                        if self._join_descriptions:
                            # 合并多个描述
                            node["description"] = "\n".join(
                                list(
                                    {
                                        *_unpack_descriptions(node),
                                        entity_description,
                                    }
                                )
                            )
                        else:
                            # 保留长的描述
                            if len(entity_description) > len(node["description"]):
                                node["description"] = entity_description
                        # 合并源文本ID
                        node["source_id"] = ", ".join(
                            list(
                                {
                                    *_unpack_source_ids(node),
                                    str(source_doc_id),
                                }
                            )
                        )
                        node["entity_type"] = (
                            entity_type if entity_type != "" else node["entity_type"]
                        )
                    else:
                        # 添加新节点
                        graph.add_node(
                            entity_name,
                            type=entity_type,
                            description=entity_description,
                            source_id=str(source_doc_id)
                        )
                # 图对象为关系
                if (
                    record_attributes[0] == '"relationship"'
                    and len(record_attributes) >= 5
                ):
                    source = clean_str(record_attributes[1].upper())
                    target = clean_str(record_attributes[2].upper())
                    edge_description = clean_str(record_attributes[3])
                    edge_source_id = clean_str(str(source_doc_id))
                    weight = (
                        float(record_attributes[-1])
                        if isinstance(record_attributes[-1], numbers.Number)
                        else 1.0
                    )
                    if source not in graph.nodes():
                        graph.add_node(
                            source,
                            type="",
                            description="",
                            source_id=edge_source_id,
                        )
                    if target not in graph.nodes():
                        graph.add_node(
                            target,
                            type="",
                            description="",
                            source_id=edge_source_id,
                        )
                    if graph.has_edge(source, target):
                        edge_data = graph.get_edge_data(source, target)
                        if edge_data is not None:
                            weight += edge_data["weight"]
                            if self._join_descriptions:
                                # 合并关系的描述
                                edge_description = "\n".join(
                                    list(
                                        {
                                            *_unpack_descriptions(edge_data),
                                            edge_description,
                                        }
                                    )
                                )
                            # 合并源文本ID
                            edge_source_id = ", ".join(
                                list(
                                    {
                                        *_unpack_source_ids(edge_data),
                                        str(source_doc_id),
                                    }
                                )
                            )
                    # 添加边
                    graph.add_edge(
                        source,
                        target,
                        weight=weight,
                        description=edge_description,
                        source_id=edge_source_id,
                    )

        return graph


def _unpack_descriptions(data: Mapping) -> list[str]:
    """
    解包节点的描述
    :param data: 节点
    :return: 节点的多个描述
    """
    value = data.get("description", None)
    return [] if value is None else value.split("\n")


def _unpack_source_ids(data: Mapping) -> list[str]:
    """
    解包源文本ID
    :param data: 节点
    :return: 节点的多个ID
    """
    value = data.get("source_id", None)
    return [] if value is None else value.split(", ")
