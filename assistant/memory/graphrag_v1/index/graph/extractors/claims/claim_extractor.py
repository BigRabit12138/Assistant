import logging
import traceback

from typing import Any
from dataclasses import dataclass

import tiktoken
import assistant.memory.graphrag_v1.config.defaults as defaults

from assistant.memory.graphrag_v1.llm import CompletionLLM
from assistant.memory.graphrag_v1.index.typing import ErrorHandlerFn

from assistant.memory.graphrag_v1.index.graph.extractors.claims.prompts import (
    LOOP_PROMPT,
    CONTINUE_PROMPT,
    CLAIM_EXTRACTION_PROMPT,
)

DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"
log = logging.getLogger(__name__)


@dataclass
class ClaimExtractorResult:
    output: list[dict]
    source_docs: dict[str, Any]


class ClaimExtractor:
    _llm: CompletionLLM
    _extraction_prompt: str
    _summary_prompt: str
    _output_formatter_prompt: str
    _input_text_key: str
    _input_entity_spec_key: str
    _input_claim_description_key: str
    _tuple_delimiter_key: str
    _record_delimiter_key: str
    _completion_delimiter_key: str
    _max_gleanings: int
    _on_error: ErrorHandlerFn

    def __init__(
            self,
            llm_invoker: CompletionLLM,
            extraction_prompt: str | None = None,
            input_text_key: str | None = None,
            input_entity_spec_key: str | None = None,
            input_claim_description_key: str | None = None,
            input_resolved_entities_key: str | None = None,
            tuple_delimiter_key: str | None = None,
            record_delimiter_key: str | None = None,
            completion_delimiter_key: str | None = None,
            encoding_model: str | None = None,
            max_gleanings: int | None = None,
            on_error: ErrorHandlerFn | None = None,
    ):
        self._llm = llm_invoker
        self._extraction_prompt = extraction_prompt or CLAIM_EXTRACTION_PROMPT
        self._input_text_key = input_text_key or "input_text"
        self._input_entity_spec_key = input_entity_spec_key or "entity_specs"
        self._tuple_delimiter_key = tuple_delimiter_key or "tuple_delimiter"
        self._record_delimiter_key = record_delimiter_key or "record_delimiter"
        self._completion_delimiter_key = (
            completion_delimiter_key or "completion_delimiter"
        )
        self._input_claim_description_key = (
            input_claim_description_key or "claim_description"
        )
        self._input_resolved_entities_key = (
            input_resolved_entities_key or "resolved_entities"
        )
        self._max_gleanings = (
            max_gleanings if max_gleanings is not None else defaults.CLAIM_MAX_GLEANINGS
        )
        self._on_error = on_error or (lambda _e, _s, _d: None)

        encoding = tiktoken.get_encoding(encoding_model or "cl100k_base")
        yes = encoding.encode("YES")
        no = encoding.encode("NO")
        self._loop_args = {"logit_bias": {yes[0]: 100, no[0]: 100}, "max_tokens": 1}

    async def __call__(
            self,
            inputs: dict[str, Any],
            prompt_variables: dict | None = None
    ) -> ClaimExtractorResult:
        if prompt_variables is None:
            prompt_variables = {}

        texts = inputs[self._input_text_key]
        entity_spec = str(inputs[self._input_entity_spec_key])
        claim_description = inputs[self._input_claim_description_key]
        resolved_entities = inputs.get(self._input_resolved_entities_key, {})
        source_doc_map = {}

        prompt_args = {
            self._input_entity_spec_key: entity_spec,
            self._input_claim_description_key: claim_description,
            self._tuple_delimiter_key: prompt_variables.get(self._tuple_delimiter_key)
            or DEFAULT_TUPLE_DELIMITER,
            self._record_delimiter_key: prompt_variables.get(self._record_delimiter_key)
            or DEFAULT_RECORD_DELIMITER,
            self._completion_delimiter_key: prompt_variables.get(
                self._completion_delimiter_key
            )
            or DEFAULT_COMPLETION_DELIMITER,
        }

        all_claims: list[dict] = []
        for doc_index, text in enumerate(texts):
            document_id = f"d{doc_index}"
            try:
                claims = await self._process_document(
                    prompt_args,
                    text,
                    doc_index
                )
                all_claims += [
                    self._clean_claim(
                        c,
                        document_id,
                        resolved_entities
                    )
                    for c in claims
                ]
                source_doc_map[document_id] = text
            except Exception as e:
                log.exception("error extracting claim.")
                self._on_error(
                    e,
                    traceback.format_exc(),
                    {"doc_index": doc_index, "text": text},
                )
                continue

        return ClaimExtractorResult(
            output=all_claims,
            source_docs=source_doc_map,
        )

    @staticmethod
    def _clean_claim(
            claim: dict,
            document_id: str,
            resolved_entities: dict
    ) -> dict:
        obj = claim.get("object_id", claim.get("object"))
        subject = claim.get("subject_id", claim.get("subject"))

        obj = resolved_entities.get(obj, obj)
        subject = resolved_entities.get(subject, subject)
        claim["object_id"] = obj
        claim["subject_id"] = subject
        claim["doc_id"] = document_id
        return claim

    async def _process_document(
            self,
            prompt_args: dict,
            doc,
            doc_index: int
    ) -> list[dict]:
        record_delimiter = prompt_args.get(
            self._record_delimiter_key, DEFAULT_RECORD_DELIMITER
        )
        completion_delimiter = prompt_args.get(
            self._completion_delimiter_key, DEFAULT_COMPLETION_DELIMITER
        )

        response = await self._llm(
            self._extraction_prompt,
            variables={
                self._input_text_key: doc,
                **prompt_args,
            }
        )
        results = response.output or ""
        claims = results.strip().removesuffix(completion_delimiter)

        for i in range(self._max_gleanings):
            glean_response = await self._llm(
                CONTINUE_PROMPT,
                name=f"extract-continuation-{i}",
                history=response.history or [],
            )
            extension = glean_response.output or ""
            claims += record_delimiter + extension.strip().removesuffix(
                completion_delimiter
            )

            if i >= self._max_gleanings - 1:
                break

            continue_response = await self._llm(
                LOOP_PROMPT,
                name=f"extract-loopcheck-{i}",
                history=glean_response.history or [],
                model_parameters=self._loop_args,
            )
            if continue_response.output != "YES":
                break

        result = self._parse_claim_tuples(results, prompt_args)
        for r in result:
            r["doc_id"] = f"{doc_index}"
        return result

    def _parse_claim_tuples(
            self,
            claims: str,
            prompt_variables: dict
    ) -> list[dict[str, Any]]:
        record_delimiter = prompt_variables.get(
            self._record_delimiter_key, DEFAULT_RECORD_DELIMITER
        )
        completion_delimiter = prompt_variables.get(
            self._completion_delimiter_key, DEFAULT_COMPLETION_DELIMITER
        )
        tuple_delimiter = prompt_variables.get(
            self._tuple_delimiter_key, DEFAULT_TUPLE_DELIMITER
        )

        def pull_field(
                index: int,
                fields: list[str]
        ) -> str | None:
            return fields[index].strip() if len(fields) > index else None

        result: list[dict[str, Any]] = []
        claims_values = (
            claims.strip().removesuffix(completion_delimiter).split(record_delimiter)
        )
        for claim in claims_values:
            claim = claim.strip().removeprefix("(").removesuffix(")")

            if claim == completion_delimiter:
                continue

            claim_fields = claim.split(tuple_delimiter)
            result.append(
                {
                    "subject_id": pull_field(0, claim_fields),
                    "object_id": pull_field(1, claim_fields),
                    "type": pull_field(2, claim_fields),
                    "status": pull_field(3, claim_fields),
                    "start_date": pull_field(4, claim_fields),
                    "end_date": pull_field(5, claim_fields),
                    "description": pull_field(6, claim_fields),
                    "source_text": pull_field(7, claim_fields),
                    "doc_id": pull_field(8, claim_fields),
                }
            )
        return result
