import os
import cmd
os.environ["AIO_NUM_THREADS"]="32"

import torch
torch.set_num_threads(32)

# make sure model is available
model_path="llama-2-7b-chat.Q4_K_M.gguf"
if not os.path.isfile(model_path):
    print("Model: ", model_path ," is not available")
    print("Please download the model in current folder")
    quit()
# base reference code : https://docs.llamaindex.ai/en/stable/examples/low_level/oss_ingestion_retrieval/
# changed vectored database to chromadb
# changed document loaded to SimpleDirectoryReader

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.vector_stores.chroma import ChromaVectorStore
from typing import Optional, Any, List
import chromadb

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
embed_model._model = torch.compile(embed_model._model, backend='aio', options={"modelname": "BAAI/bge-small-en"})

llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    model_url=None,
    # OR you can set the path to a pre-downloaded model instead of model_url
    model_path=model_path,
    temperature=0.7,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    verbose=False,
)

#load documents from data folder
documents = SimpleDirectoryReader("./data/").load_data()

db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

query_str = "This is a test question?"
query_embedding = embed_model.get_query_embedding(query_str)

query_mode = "default"
# query_mode = "sparse"
# query_mode = "hybrid"

# vcector store
vector_store_query = VectorStoreQuery(
    query_embedding=query_embedding, similarity_top_k=2, mode=query_mode
)

# retriver
class VectorDBRetriever(BaseRetriever):
    """Retriever over a postgres vector store."""
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        """Init params."""
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        query_embedding = embed_model.get_query_embedding(
            query_bundle.query_str
        )
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = vector_store.query(vector_store_query)
        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))
        return nodes_with_scores

retriever = VectorDBRetriever(
    vector_store, embed_model, query_mode="default", similarity_top_k=2
)
query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm, streaming=True)

# CLI to QA
class LlmCLI(cmd.Cmd):
    prompt = 'llama QA >> '
    intro = """Welcome to llama CLI. Reserved first words help, upload and quit
               Type "help" for available commands."""

    def default(self, line):
        """Default """
        # invoke chain with question asked
        streaming_response = query_engine.query(line)
        for text in streaming_response.response_gen:
            print(text, end='', flush=True)
        print()

    def do_quit(self, line):
        """Exit the CLI."""
        return True
        
    def do_upload(self, line):
        """Upload text file into vectordb e.g. >> upload some_file.txt"""
        if os.path.isfile(line):
            print("Uploading ", line, end="")
            documents = SimpleDirectoryReader(input_files=[line]).load_data()
            index = VectorStoreIndex.from_documents(
                    documents, storage_context=storage_context, embed_model=embed_model
                    )
            print(" [Done]")
        else:
            print("File ", line,  "not found")

if __name__ == '__main__':
    LlmCLI().cmdloop()

