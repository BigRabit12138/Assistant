import os.path

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

import global_var


class ChromaDB:
    def __init__(self):
        self.embedding = SentenceTransformerEmbeddings(model_name=global_var.embedding_function,
                                                       model_kwargs={'device': 'cuda'})
        self.persist_dir = os.path.join(global_var.project_dir, 'Memory/ChromaDB')
        self.chroma = Chroma(embedding_function=self.embedding,
                             persist_directory=self.persist_dir)
