class ApiKeyMissingError(ValueError):
    def __init__(self, embedding: bool = False) -> None:
        api_type = "Embedding" if embedding else "Completion"
        api_key = "GRAPHRAG_EMBEDDING_API_KEY" if embedding else "GRAPHRAG_LLM_API_KEY"
        msg = f"API Key is required for {api_type} API. Please set either the OPENAI_API_KEY, \
        GRAPHRAG_API_KEY or {api_key} environment variable."
        super().__init__(msg)


class AzureApiBaseMissingError(ValueError):
    def __init__(self, embedding: bool = False) -> None:
        api_type = 'Embedding' if embedding else "Completion"
        api_base = "GRAPHRAG_EMBEDDING_API_BASE" if embedding else "GRAPHRAG_API_BASE"
        msg = f"API Base is required for {api_type} API. Please set either the OPENAI_API_BASE, GRAPHRAG_API_BASE \
        or {api_base} environment variable."
        super().__init__(msg)


class AzureDeploymentNameMissingError(ValueError):
    def __init__(self, embedding: bool = False) -> None:
        api_type = "Embedding" if embedding else "Completion"
        api_base = (
            "GRAPHRAG_EMBEDDING_DEPLOYMENT_NAME"
            if embedding else "GRAPHRAG_LLM_DEPLOYMENT_NAME"
        )
        msg = f"Deployment Name is required for {api_type} API. Please set either the OPENAI_DEPLOYMENT_NAME, \
        GRAPHRAG_LLM_DEPLOYMENT_NAME or {api_base} environment variable."
        super().__init__(msg)
