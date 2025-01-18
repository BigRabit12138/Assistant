import gc
import json
import time
import logging
import traceback

from io import BytesIO
from typing import cast
from pathlib import Path
from string import Template
from dataclasses import asdict
from collections.abc import AsyncIterable

import pandas as pd

from datashaper import (
    Workflow,
    MemoryProfile,
    WorkflowCallbacks,
    WorkflowRunResult,
    DEFAULT_INPUT_NAME,
    WorkflowCallbacksManager,
)

from assistant.memory.graphrag_v1.index.cache import (
    load_cache,
    InMemoryCache,
    PipelineCache,
)
from assistant.memory.graphrag_v1.index.config import (
    PipelineConfig,
    PipelineWorkflowStep,
    PipelineFileCacheConfig,
    PipelineBlobCacheConfig,
    PipelineInputConfigTypes,
    PipelineCacheConfigTypes,
    PipelineMemoryCacheConfig,
    PipelineFileStorageConfig,
    PipelineBlobStorageConfig,
    PipelineWorkflowReference,
    PipelineStorageConfigTypes,
    PipelineFileReportingConfig,
    PipelineBlobReportingConfig,
    PipelineReportingConfigTypes,
)
from assistant.memory.graphrag_v1.index.context import (
    PipelineRunStats,
    PipelineRunContext,
)
from assistant.memory.graphrag_v1.index.emit import (
    TableEmitterType,
    create_table_emitters,
)
from assistant.memory.graphrag_v1.index.input import load_input
from assistant.memory.graphrag_v1.index.load_pipeline_config import load_pipeline_config
from assistant.memory.graphrag_v1.index.progress import (
    ProgressReporter,
    NullProgressReporter,
)
from assistant.memory.graphrag_v1.index.reporting import (
    load_pipeline_reporter,
    ConsoleWorkflowCallbacks,
    ProgressWorkflowCallbacks,
)
from assistant.memory.graphrag_v1.index.storage import (
    load_storage,
    PipelineStorage,
    MemoryPiplineStorage,
)
from assistant.memory.graphrag_v1.index.typing import PipelineRunResult
from assistant.memory.graphrag_v1.index.verbs import *
from assistant.memory.graphrag_v1.index.workflows import (
    load_workflows,
    create_workflow,
    VerbDefinitions,
    WorkflowDefinitions,
)

log = logging.getLogger(__name__)


async def run_pipeline_with_config(
        config_or_path: PipelineConfig | str,
        workflows: list[PipelineWorkflowReference] | None = None,
        dataset: pd.DataFrame | None = None,
        storage: PipelineStorage | None = None,
        cache: PipelineCache | None = None,
        callbacks: WorkflowCallbacks | None = None,
        progress_reporter: ProgressReporter | None = None,
        input_post_process_steps: list[PipelineWorkflowStep] | None = None,
        additional_verbs: VerbDefinitions | None = None,
        additional_workflows: WorkflowDefinitions | None = None,
        emit: list[TableEmitterType] | None = None,
        memory_profile: bool = False,
        run_id: str | None = None,
        is_resume_run: bool = False,
        **_kwargs: dict,
) -> AsyncIterable[PipelineRunResult]:
    if isinstance(config_or_path, str):
        log.info(f"Running pipeline with config {config_or_path}.")
    else:
        log.info("Running pipeline.")

    run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
    config = load_pipeline_config(config_or_path)
    config = _apply_substitutions(config, run_id)
    root_dir = config.root_dir

    def _create_storage(config_: PipelineStorageConfigTypes | None) -> PipelineStorage:
        return load_storage(
            config_
            or PipelineFileStorageConfig(base_dir=str(Path(root_dir or "") / "output"))
        )

    def _create_cache(config_: PipelineCacheConfigTypes | None) -> PipelineCache:
        return load_cache(config_ or PipelineMemoryCacheConfig(), root_dir=root_dir)

    def _create_reporter(
            config_: PipelineReportingConfigTypes | None,
    ) -> WorkflowCallbacks | None:
        return load_pipeline_reporter(config_, root_dir) if config_ else None

    async def _create_input(
            config_: PipelineInputConfigTypes | None,
    ) -> pd.DataFrame | None:
        if config_ is None:
            return None

        return await load_input(config_, progress_reporter, root_dir)

    def _create_postprocess_steps(
            config_: PipelineInputConfigTypes | None,
    ) -> list[PipelineWorkflowStep] | None:
        return config_.post_process if config_ is not None else None

    progress_reporter = progress_reporter or NullProgressReporter()
    storage = storage or _create_storage(config.storage)
    cache = cache or _create_cache(config.cache)
    callbacks = callbacks or _create_reporter(config.reporting)
    dataset = dataset if dataset is not None else await _create_input(config.input)
    post_process_steps = input_post_process_steps or _create_postprocess_steps(config.input)
    workflows = workflows or config.workflows

    if dataset is None:
        msg = "No dataset provided!"
        raise ValueError(msg)

    async for table in run_pipeline(
        workflows=workflows,
        dataset=dataset,
        storage=storage,
        cache=cache,
        callbacks=callbacks,
        input_post_process_steps=post_process_steps,
        memory_profile=memory_profile,
        additional_verbs=additional_verbs,
        additional_workflows=additional_workflows,
        progress_reporter=progress_reporter,
        emit=emit,
        is_resume_run=is_resume_run,
    ):
        yield table


async def run_pipeline(
        workflows: list[PipelineWorkflowReference],
        dataset: pd.DataFrame,
        storage: PipelineStorage | None = None,
        cache: PipelineCache | None = None,
        callbacks: WorkflowCallbacks | None = None,
        progress_reporter: ProgressReporter | None = None,
        input_post_process_steps: list[PipelineWorkflowStep] | None = None,
        additional_verbs: VerbDefinitions | None = None,
        additional_workflows: WorkflowDefinitions | None = None,
        emit: list[TableEmitterType] | None = None,
        memory_profile: bool = False,
        is_resume_run: bool = False,
        **_kwargs: dict,
) -> AsyncIterable[PipelineRunResult]:
    start_time = time.time()
    stats = PipelineRunStats()
    storage = storage or MemoryPiplineStorage()
    cache = cache or InMemoryCache()
    progress_reporter = progress_reporter or NullProgressReporter()
    callbacks = callbacks or ConsoleWorkflowCallbacks()
    callbacks = _create_callback_chain(callbacks, progress_reporter)
    emit = emit or [TableEmitterType.Parquet]
    emitters = create_table_emitters(
        emit,
        storage,
        lambda e_, s, d: cast(WorkflowCallbacks, callbacks).on_error(
            "Error emitting table", e_, s, d
        ),
    )
    loaded_workflows = load_workflows(
        workflows,
        additional_verbs=additional_verbs,
        additional_workflows=additional_workflows,
        memory_profile=memory_profile,
    )
    workflows_to_run = loaded_workflows.workflows
    workflow_dependencies = loaded_workflows.dependencies

    context = _create_run_context(storage, cache, stats)

    if len(emitters) == 0:
        log.info(
            "No emitters provided. No table outputs will be generated. This is probably not correct."
        )

    async def dump_stats() -> None:
        await storage.set("stats.json", json.dumps(asdict(stats), indent=4))

    async def load_table_from_storage(name: str) -> pd.DataFrame:
        if not await storage.has(name):
            msg = f"Could not find {name} in storage!"
            raise ValueError(msg)
        try:
            log.info(f"read table form storage: {name}.")
            return pd.read_parquet(BytesIO(await storage.get(name, as_bytes=True)))
        except Exception:
            log.exception(f"error loading table from storage: {name}.")
            raise

    async def inject_workflow_data_dependencies(workflow_: Workflow) -> None:
        workflow_.add_table(DEFAULT_INPUT_NAME, dataset)
        deps = workflow_dependencies[workflow_.name]
        log.info(f"dependencies for {workflow_.name}: {deps}.")
        for id_ in deps:
            workflow_id = f"workflow:{id_}"
            table = await load_table_from_storage(f"{id_}.parquet")
            workflow_.add_table(workflow_id, table)

    async def write_workflow_stats(
            workflow_: Workflow,
            workflow_result: WorkflowRunResult,
            workflow_start_time_: float,
    ) -> None:
        for vt in workflow_result.verb_timings:
            stats.workflows[workflow_.name][f"{vt.index}_{vt.verb}"] = vt.timing

        workflow_end_time = time.time()
        stats.workflows[workflow_.name]["overall"] = (
                workflow_end_time - workflow_start_time_
        )
        stats.total_runtime = time.time() - start_time
        await dump_stats()

        if workflow_result.memory_profile is not None:
            await _save_profiler_stats(
                storage,
                workflow_.name,
                workflow_result.memory_profile,
            )
        log.debug(
            f"first row of {workflow_name} => {workflow_.output().iloc[0].to_json()}."
        )

    async def emit_workflow_output(workflow_: Workflow) -> pd.DataFrame:
        output_ = cast(pd.DataFrame, workflow_.output())
        for emitter in emitters:
            await emitter.emit(workflow_.name, output_)
        return output_

    dataset = await _run_post_process_steps(
        input_post_process_steps,
        dataset,
        context,
        callbacks
    )

    _validate_dataset(dataset)

    log.info(f"Final # of rows loaded: {len(dataset)}.")
    stats.num_documents = len(dataset)
    last_workflow = "input"

    try:
        await dump_stats()

        for workflow_to_run in workflows_to_run:
            gc.collect()

            workflow = workflow_to_run.workflow
            workflow_name: str = workflow.name
            last_workflow = workflow_name

            log.info(f"Running workflow: {workflow_name}...")

            if is_resume_run and await storage.has(
                f"{workflow_to_run.workflow.name}.parquet"
            ):
                log.info(f"Skipping {workflow_name} because it already exists.")
                continue

            stats.workflows[workflow_name] = {"overall": 0.0}
            await inject_workflow_data_dependencies(workflow)

            workflow_start_time = time.time()
            result = await workflow.run(context, callbacks)
            await write_workflow_stats(workflow, result, workflow_start_time)

            output = await emit_workflow_output(workflow)
            yield PipelineRunResult(workflow_name, output, None)
            del output
            workflow.dispose()
            del workflow

        stats.total_runtime = time.time() - start_time
        await dump_stats()
    except Exception as e:
        log.exception(
            f"error running workflow {last_workflow}."
        )
        cast(WorkflowCallbacks, callbacks).on_error(
            "Error running pipeline!", e, traceback.format_exc()
        )
        yield PipelineRunResult(last_workflow, None, [e])


def _create_callback_chain(
        callbacks: WorkflowCallbacks | None,
        progress: ProgressReporter | None,
) -> WorkflowCallbacks:
    manager = WorkflowCallbacksManager()
    if callbacks is not None:
        manager.register(callbacks)
    if progress is not None:
        manager.register(ProgressWorkflowCallbacks(progress))
    return manager


async def _save_profiler_stats(
        storage: PipelineStorage,
        workflow_name: str,
        profile: MemoryProfile,
):
    await storage.set(
        f"{workflow_name}_profiling.peak_stats.csv",
        profile.peak_stats.to_csv(index=True),
    )

    await storage.set(
        f"{workflow_name}_profiling.snapshot_status.csv",
        profile.snapshot_stats.to_csv(index=True),
    )

    await storage.set(
        f"{workflow_name}_profiling.time_stats.csv",
        profile.time_stats.to_csv(index=True),
    )

    await storage.set(
        f"{workflow_name}_profiling.detailed_view.csv",
        profile.detailed_view.to_csv(index=True),
    )


async def _run_post_process_steps(
        post_process: list[PipelineWorkflowStep] | None,
        dataset: pd.DataFrame,
        context: PipelineRunContext,
        callbacks: WorkflowCallbacks,
) -> pd.DataFrame:
    if post_process is not None and len(post_process) > 0:
        input_workflow = create_workflow(
            "Input Post Process",
            post_process,
        )
        input_workflow.add_table(DEFAULT_INPUT_NAME, dataset)
        await input_workflow.run(
            context=context,
            callbacks=callbacks,
        )
        dataset = cast(pd.DataFrame, input_workflow.output())
    return dataset


def _validate_dataset(dataset: pd.DataFrame):
    if not isinstance(dataset, pd.DataFrame):
        msg = "Dataset must be a pandas dataframe!"
        raise TypeError(msg)


def _apply_substitutions(
        config: PipelineConfig,
        run_id: str
) -> PipelineConfig:
    substitutions = {"timestamp": run_id}

    if (
        isinstance(
            config.storage, PipelineFileStorageConfig | PipelineBlobStorageConfig
        )
        and config.storage.base_dir
    ):
        config.storage.base_dir = Template(config.storage.base_dir).substitute(
            substitutions
        )
    if (
        isinstance(config.cache, PipelineFileCacheConfig | PipelineBlobCacheConfig)
        and config.cache.base_dir
    ):
        config.cache.base_dir = Template(config.cache.base_dir).substitute(
            substitutions
        )

    if (
        isinstance(
            config.reporting, PipelineFileReportingConfig | PipelineBlobReportingConfig
        )
        and config.reporting.base_dir
    ):
        config.reporting.base_dir = Template(config.reporting.base_dir).substitute(
            substitutions
        )

    return config


def _create_run_context(
        storage: PipelineStorage,
        cache: PipelineCache,
        stats: PipelineRunStats,
) -> PipelineRunContext:
    return PipelineRunContext(
        stats=stats,
        cache=cache,
        storage=storage,
    )
