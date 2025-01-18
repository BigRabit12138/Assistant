from assistant.memory.graphrag_v1.index.text_splitting.text_splitting import (
    TokenTextSplitter
)


def check_token_limit(text, max_token):
    text_splitter = TokenTextSplitter(
        chunk_size=max_token,
        chunk_overlap=0
    )
    docs = text_splitter.split_text(text)
    if len(docs) > 1:
        return 0
    return 1
