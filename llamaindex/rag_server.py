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

import torch
torch.set_num_threads(32)

from fastapi import FastAPI, File, UploadFile
import uvicorn
import asyncio
import os

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
embed_model._model = torch.compile(embed_model._model, backend='aio', options={"modelname": "BAAI/bge-small-en"})

model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"

llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    model_url=model_url,
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 0},
    verbose=True,
)

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

# construct vector store query

query_mode = "default"
# query_mode = "sparse"
# query_mode = "hybrid"

vector_store_query = VectorStoreQuery(
    query_embedding=query_embedding, similarity_top_k=2, mode=query_mode
)

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
query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)

app = FastAPI()

# update running to some good status and service at the startup of file
@app.get("/")
async def read_root():
    return { "LlamaIndex RAG" : "Running" }

@app.post("/upload-file/")
async def create_upload_file(uploaded_file: UploadFile = File(...)):
    file_location = f"data/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())
    documents = SimpleDirectoryReader(input_files=[file_location]).load_data()
    index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, embed_model=embed_model
    )
    os.remove(file_location)
    return {"info": f"file '{uploaded_file.filename}' moved into vectordb"}

@app.post("/question/")
async def question_answer(question: str):
    response = query_engine.query(question)
    return {"question:": f"question asked is {question} . answer {response}"}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
