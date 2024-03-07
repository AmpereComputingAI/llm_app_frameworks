from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from llama_index.core import set_global_tokenizer
from transformers import AutoTokenizer

import weaviate
client = weaviate.Client("http://localhost:8080")

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
    generate_kwargs={},
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
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# load documents
print("=== Loading Documents....")
documents = SimpleDirectoryReader(
    "./data/articles/"
).load_data()

# create vector store index
print("=== Creating Vector Index....")

vector_store = WeaviateVectorStore(weaviate_client=client, index_name="LlamaIndex")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model, storage_context=storage_context)

# set up query engine
print("=== Instantiating Query Engine....")
query_engine = index.as_query_engine(llm=llm)

query_input = "What did the author do growing up?"
print("=== Querying LLM: {}...".format(query_input))
response = query_engine.query(query_input)
print(response)

query_input = "What did the president say about Putin in the state of the union?"
print("=== Querying LLM: {}...".format(query_input))
response = query_engine.query(query_input)
print(response)

