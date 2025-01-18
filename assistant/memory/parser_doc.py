import os

from typing import List
from transformers import BertTokenizer

from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


import global_var

from Memory.chroma import ChromaDB


class ParserDoc:
    def __init__(self):
        self.chroma = ChromaDB().chroma
        self.tokenizer = BertTokenizer.from_pretrained(global_var.embedding_function)
        self.splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(self.tokenizer,
                                                                                  chunk_size=global_var.chunk_size)

    @staticmethod
    def parser(file_path: str) -> List[Document]:
        assert os.path.exists(file_path), 'file does not exist.'
        file_name = os.path.basename(file_path)
        name, ext = os.path.splitext(file_name)

        if ext.lower() == '.txt':
            doc = TextLoader(file_path=file_path, autodetect_encoding=True).load()
            return doc
        elif ext.lower() == '.pdf':
            pass
        elif ext.lower() == '.tex':
            pass
        elif ext.lower() == '.doc' or ext.lower() == '.docx':
            pass
        elif ext.lower() == '.pptx':
            pass
        elif ext.lower() == '.xlsx':
            pass
        elif ext.lower() == '.md' or ext.lower() == '.markdown':
            pass
        elif ext.lower() == '.epub':
            pass
        elif ext.lower() == '.mobi':
            pass
        else:
            raise ValueError('file type does not support, document should be one of'
                             ' txt, pdf, tex, doc, docx, ppt, xlsx, md, epub, mobi')

    def parser_and_store(self, file_path: str) -> None:
        docs = ParserDoc.parser(file_path)
        for doc in docs:
            assert doc.page_content != ''
        docs = self.splitter.split_documents(docs)
        self.chroma.add_documents(docs)
