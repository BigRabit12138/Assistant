from typing import cast
from pathlib import Path

from datashaper import WorkflowCallbacks

from assistant.memory.graphrag_v1.config import ReportingType
from assistant.memory.graphrag_v1.index.config import (
    PipelineReportingConfig,
    PipelineFileReportingConfig,
    PipelineBlobReportingConfig
)
from assistant.memory.graphrag_v1.index.reporting.blob_workflow_callbacks import (
    BlobWorkflowCallbacks
)
from assistant.memory.graphrag_v1.index.reporting.file_workflow_callbacks import (
    FileWorkflowCallbacks
)
from assistant.memory.graphrag_v1.index.reporting.console_workflow_callbacks import (
    ConsoleWorkflowCallbacks
)


def load_pipeline_reporter(
        config: PipelineReportingConfig | None,
        root_dir: str | None
) -> WorkflowCallbacks:
    config = config or PipelineFileReportingConfig(base_dir="reports")

    match config.type:
        case ReportingType.file:
            config = cast(PipelineFileReportingConfig, config)
            return FileWorkflowCallbacks(
                str(Path(root_dir or "") / (config.base_dir or ""))
            )
        case ReportingType.console:
            return ConsoleWorkflowCallbacks()
        case ReportingType.blob:
            config = cast(PipelineBlobReportingConfig, config)
            return BlobWorkflowCallbacks(
                config.connection_string,
                config.container_name,
                base_dir=config.base_dir,
                storage_account_blob_url=config.storage_account_blob_url
            )
        case _:
            msg = f"Unknown reporting type: {config.type}."
            raise ValueError(msg)
