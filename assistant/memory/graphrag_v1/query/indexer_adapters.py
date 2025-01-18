from typing import cast

import pandas as pd

from assistant.memory.graphrag_v1.model import (
    Entity,
    TextUnit,
    Covariate,
    Relationship,
    CommunityReport,
)
from assistant.memory.graphrag_v1.query.input.loaders.dfs import (
    read_entities,
    read_text_units,
    read_covariates,
    read_relationships,
    read_community_reports,
)


def read_indexer_text_units(
        final_text_units: pd.DataFrame
) -> list[TextUnit]:
    return read_text_units(
        df=final_text_units,
        short_id_col=None,
        covariates_col=None,
    )


def read_indexer_covariates(final_covariates: pd.DataFrame) -> list[Covariate]:
    covariate_df = final_covariates
    covariate_df["id"] = covariate_df["id"].astype(str)
    return read_covariates(
        df=covariate_df,
        short_id_col="human_readable_id",
        attributes_cols=[
            "object_id",
            "status",
            "start_data",
            "end_data",
            "description",
        ],
        text_unit_ids_col=None,
    )


def read_indexer_relationships(final_relationships: pd.DataFrame) -> list[Relationship]:
    return read_relationships(
        df=final_relationships,
        short_id_col="human_readable_id",
        description_embedding_col=None,
        document_ids_col=None,
        attributes_cols=["rank"],
    )


def read_indexer_reports(
        final_community_reports: pd.DataFrame,
        final_nodes: pd.DataFrame,
        community_level: int,
) -> list[CommunityReport]:
    report_df = final_community_reports
    entity_df = final_nodes
    entity_df = _filter_under_community_level(entity_df, community_level)
    entity_df.loc[:, "community"] = entity_df["community"].fillna(-1)
    entity_df.loc[:, "community"] = entity_df["community"].astype(int)

    entity_df = entity_df.groupby(['title']).agg({"community": "max"}).reset_index()
    entity_df["community"] = entity_df["community"].astype(str)
    filtered_community_df = entity_df["community"].drop_duplicates()

    report_df = _filter_under_community_level(report_df, community_level)
    report_df = report_df.merge(filtered_community_df, on="community", how="inner")

    return read_community_reports(
        df=report_df,
        id_col="community",
        short_id_col="community",
        summary_embedding_col=None,
        content_embedding_col=None,
    )


def read_indexer_entities(
        final_nodes: pd.DataFrame,
        final_entities: pd.DataFrame,
        community_level: int,
) -> list[Entity]:
    entity_df = final_nodes
    entity_embedding_df = final_entities

    entity_df = _filter_under_community_level(entity_df, community_level)
    entity_df = cast(pd.DataFrame, entity_df[["title", "degree", "community"]]).rename(
        columns={"title": "name", "degree": "rank"}
    )

    entity_df["community"] = entity_df["community"].fillna(-1)
    entity_df["community"] = entity_df["community"].astype(int)
    entity_df["rank"] = entity_df["rank"].astype(int)

    entity_df = (
        entity_df.groupby(["name", "rank"]).agg({"community": "max"}).reset_index()
    )
    entity_df["community"] = entity_df["community"].apply(lambda x: [str(x)])
    entity_df = entity_df.merge(
        entity_embedding_df, on="name", how="inner"
    ).drop_duplicates(subset=["name"])

    return read_entities(
        df=entity_df,
        id_col="id",
        title_col="name",
        type_col="type",
        short_id_col="human_readable_id",
        description_col="description",
        community_col="community",
        rank_col="rank",
        name_embedding_col=None,
        description_embedding_col="description_embedding",
        graph_embedding_col=None,
        text_unit_ids_col="text_unit_ids",
        document_ids_col=None,
    )


def _filter_under_community_level(
        df: pd.DataFrame, community_level: int
) -> pd.DataFrame:
    return cast(
        pd.DataFrame,
        df[df.level <= community_level],
    )
