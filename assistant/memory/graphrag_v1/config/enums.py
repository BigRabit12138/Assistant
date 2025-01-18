from enum import Enum


class CacheType(str, Enum):
    file = "file"
    memory = "memory"
    none = "none"
    blob = "blob"

    def __repr__(self):
        return f'"{self.value}"'


class InputFileType(str, Enum):
    csv = 'csv'
    text = 'text'

    def __repr__(self):
        return f'"{self.value}"'


class InputType(str, Enum):
    file = "file"
    blob = 'blob'

    def __repr__(self):
        return f'"{self.value}"'


class StorageType(str, Enum):
    file = "file"
    memory = "memory"
    blob = "blob"

    def __repr__(self):
        return f'"{self.value}"'


class ReportingType(str, Enum):
    file = "file"
    console = 'console'
    blob = 'blob'

    def __repr__(self):
        return f'"{self.value}"'


class TextEmbeddingTarget(str, Enum):
    all = 'all'
    required = 'required'

    def __repr__(self):
        return f'"{self.value}"'


class LLMType(str, Enum):
    OpenAIEmbedding = 'openai_embedding'
    AzureOpenAIEmbedding = 'azure_openai_embedding'

    OpenAI = 'openai'
    AzureOpenAI = 'azure_openai'

    OpenAIChat = "openai_chat"
    AzureOpenAIChat = 'azure_openai_chat'

    StaticResponse = 'static_response'

    def __repr__(self):
        return f'"{self.value}"'
