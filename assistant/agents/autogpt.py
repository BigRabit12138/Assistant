import faiss

from langchain.agents import Tool
from langchain.utilities import SerpAPIWrapper
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools.file_management.read import ReadFileTool
from langchain.tools.file_management.write import WriteFileTool
from langchain_experimental.autonomous_agents import AutoGPT
from langchain.chat_models import ChatOpenAI

from assistant.memory import ChromaDB


class AutoGPTAgent:
    def __init__(self):
        search = SerpAPIWrapper()
        tools = [
            Tool(
                name="search",
                func=search.run,
                description="useful for when you need to answer questions about current events. "
                            "You should ask targeted questions"
            ),
            WriteFileTool(),
            ReadFileTool(),
        ]
        embeddings_model = OpenAIEmbeddings()
        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), index_to_docstore_id={})
        self.agent = AutoGPT.from_llm_and_tools(
            ai_name="Tom",
            ai_role="Assistant",
            tools=tools,
            llm=ChatOpenAI(temperature=0),
            memory=vectorstore.as_retriever(),
        )
        # Set verbose to be true
        self.agent.chain.verbose = True

    def run(self, prompts: list[str]):
        return self.agent.run(goals=prompts)

