from assistant.memory.graphrag_v1.index.workflows.typing import WorkflowDefinitions
from assistant.memory.graphrag_v1.index.workflows.v1.create_base_documents import (
    build_steps as build_create_base_documents_steps,
)
from assistant.memory.graphrag_v1.index.workflows.v1.create_base_documents import (
    workflow_name as create_base_documents,
)
from assistant.memory.graphrag_v1.index.workflows.v1.create_base_entity_graph import (
    build_steps as build_create_base_entity_graph_steps,
)
from assistant.memory.graphrag_v1.index.workflows.v1.create_base_entity_graph import (
    workflow_name as create_base_entity_graph,
)
from assistant.memory.graphrag_v1.index.workflows.v1.create_base_extracted_entities import (
    build_steps as build_create_base_extracted_entities_steps,
)
from assistant.memory.graphrag_v1.index.workflows.v1.create_base_extracted_entities import (
    workflow_name as create_base_extracted_entities,
)
from assistant.memory.graphrag_v1.index.workflows.v1.create_base_text_units import (
    build_steps as build_create_base_text_units_steps,
)
from assistant.memory.graphrag_v1.index.workflows.v1.create_base_text_units import (
    workflow_name as create_base_text_units,
)
from assistant.memory.graphrag_v1.index.workflows.v1.create_final_communities import (
    build_steps as build_create_final_communities_steps,
)
from assistant.memory.graphrag_v1.index.workflows.v1.create_final_communities import (
    workflow_name as create_final_communities,
)
from assistant.memory.graphrag_v1.index.workflows.v1.create_final_community_reports import (
    build_steps as build_create_final_community_reports_steps,
)
from assistant.memory.graphrag_v1.index.workflows.v1.create_final_community_reports import (
    workflow_name as create_final_community_reports,
)
from assistant.memory.graphrag_v1.index.workflows.v1.create_final_covariates import (
    build_steps as build_create_final_covariates_steps
)
from assistant.memory.graphrag_v1.index.workflows.v1.create_final_covariates import (
    workflow_name as create_final_covariates,
)
from assistant.memory.graphrag_v1.index.workflows.v1.create_final_documents import (
    build_steps as build_create_final_documents_steps,
)
from assistant.memory.graphrag_v1.index.workflows.v1.create_final_documents import (
    workflow_name as create_final_documents,
)
from assistant.memory.graphrag_v1.index.workflows.v1.create_final_entities import (
    build_steps as build_create_final_entities_steps,
)
from assistant.memory.graphrag_v1.index.workflows.v1.create_final_entities import (
    workflow_name as create_final_entities,
)
from assistant.memory.graphrag_v1.index.workflows.v1.create_final_nodes import (
    build_steps as build_create_final_nodes_steps,
)
from assistant.memory.graphrag_v1.index.workflows.v1.create_final_nodes import (
    workflow_name as create_final_nodes,
)
from assistant.memory.graphrag_v1.index.workflows.v1.create_final_relationships import (
    build_steps as build_create_final_relationships_steps,
)
from assistant.memory.graphrag_v1.index.workflows.v1.create_final_relationships import (
    workflow_name as create_final_relationships,
)
from assistant.memory.graphrag_v1.index.workflows.v1.create_final_text_units import (
    build_steps as build_create_final_text_units_steps,
)
from assistant.memory.graphrag_v1.index.workflows.v1.create_final_text_units import (
    workflow_name as create_final_text_units,
)
from assistant.memory.graphrag_v1.index.workflows.v1.create_summarized_entities import (
    build_steps as build_create_summarized_entities_steps
)
from assistant.memory.graphrag_v1.index.workflows.v1.create_summarized_entities import (
    workflow_name as create_summarized_entities
)
from assistant.memory.graphrag_v1.index.workflows.v1.join_text_units_to_covariate_ids import (
    build_steps as join_text_units_to_covariate_steps,
)
from assistant.memory.graphrag_v1.index.workflows.v1.join_text_units_to_covariate_ids import (
    workflow_name as join_text_units_to_covariate_ids,
)
from assistant.memory.graphrag_v1.index.workflows.v1.join_text_units_to_entity_ids import (
    build_steps as join_text_units_to_entity_ids_steps,
)
from assistant.memory.graphrag_v1.index.workflows.v1.join_text_units_to_entity_ids import (
    workflow_name as join_text_units_to_entity_ids,
)
from assistant.memory.graphrag_v1.index.workflows.v1.join_text_units_to_relationship_ids import (
    build_steps as join_text_units_to_relationship_ids_steps,
)
from assistant.memory.graphrag_v1.index.workflows.v1.join_text_units_to_relationship_ids import (
    workflow_name as join_text_units_to_relationship_ids
)

default_workflow: WorkflowDefinitions = {
    create_base_extracted_entities: build_create_base_extracted_entities_steps,
    create_base_entity_graph: build_create_base_entity_graph_steps,
    create_base_text_units: build_create_base_text_units_steps,
    create_final_text_units: build_create_final_text_units_steps,
    create_final_community_reports: build_create_final_community_reports_steps,
    create_final_nodes: build_create_final_nodes_steps,
    create_final_relationships: build_create_final_relationships_steps,
    create_final_documents: build_create_final_documents_steps,
    create_final_covariates: build_create_final_covariates_steps,
    create_base_documents: build_create_base_documents_steps,
    create_final_entities: build_create_final_entities_steps,
    create_final_communities: build_create_final_communities_steps,
    create_summarized_entities: build_create_summarized_entities_steps,
    join_text_units_to_entity_ids: join_text_units_to_entity_ids_steps,
    join_text_units_to_covariate_ids: join_text_units_to_covariate_steps,
    join_text_units_to_relationship_ids: join_text_units_to_relationship_ids_steps,
}
