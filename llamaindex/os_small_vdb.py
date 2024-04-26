import time
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from llama_index.core import set_global_tokenizer
from transformers import AutoTokenizer

import chromadb
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

model_url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"

print("=== Instantiating LLM....")

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
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=False,
)

set_global_tokenizer(
    AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf").encode
)

# use Huggingface embeddings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

print("=== Loading Embedding Model....")
model_name="BAAI/bge-small-en-v1.5"
start = time.time()
embed_model = HuggingFaceEmbedding(model_name=model_name)
embed_time = time.time() - start
# load documents
print("=== Loading Documents....")
documents = SimpleDirectoryReader(
    "./data/articles/"
).load_data()

# create vector store index
print("=== Creating Vector Index....")
start = time.time()
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model, storage_context=storage_context)
vector_store_time = time.time() - start

# set up query engine
print("=== Instantiating Query Engine....")
start = time.time()
query_engine = index.as_query_engine(llm=llm)
mk_query_engine_time = time.time() - start

query_input = "What did the author do growing up?"
print("=== Querying LLM: {}...".format(query_input))
start = time.time()
response = query_engine.query(query_input)
q1_time = time.time() - start
print(response)

query_input = "What did the president say about Putin in the state of the union?"
print("=== Querying LLM: {}...".format(query_input))
start = time.time()
response = query_engine.query(query_input)
q2_time = time.time() - start
print(response)

query_input = "What did the author do growing up?"
print("=== Querying LLM: {}...".format(query_input))
start = time.time()
response = query_engine.query(query_input)
q3_time = time.time() - start
print(response)

query_input = "What did the president say about Putin in the state of the union?"
print("=== Querying LLM: {}...".format(query_input))
start = time.time()
response = query_engine.query(query_input)
q4_time = time.time() - start
print(response)


print(f'Time taken to load Embedding      {embed_time} ({model_name})')
print(f'Time taken to create vector index {vector_store_time}')
print(f'Time taken to create query engine {mk_query_engine_time} ({model_url})')
print(f'Time taken by query 1             {q1_time}')
print(f'Time taken by query 2             {q2_time}')
print(f'Time taken by query 3             {q3_time}')
print(f'Time taken by query 4             {q4_time}')

