import time
import torch
torch.set_num_threads(32)

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

model_name="BAAI/bge-small-en-v1.5"
embedding_model = HuggingFaceEmbeddings(model_name=model_name, show_progress=False)

#embedding_model.client.forward = torch.compile(embedding_model.client.forward)
embedding_model.client.forward = torch.compile(embedding_model.client.forward, backend='aio', options={"modelname": model_name})

print("===> Loading Documents ....")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
start = time.time()
#raw_documents = DirectoryLoader( "./data/", glob="**/*.txt", use_multithreading=True).load_and_split(text_splitter)
#documents = DirectoryLoader( "./data/", glob="**/*.txt", use_multithreading=True).load_and_split(text_splitter)
documents = DirectoryLoader( "./data/", use_multithreading=True).load_and_split(text_splitter)
document_load_time = time.time() - start
print("===> Chunking Documents ....")
start = time.time()
#documents = text_splitter.split_documents(raw_documents)
chunking_time = time.time() - start
print("===> Vector store ....")
start = time.time()
db = Chroma.from_documents(documents, embedding_model)
vector_store_time = time.time() - start

query = "What did the president say assault weapon"
print("===> Run Query: {} ".format(query))
start = time.time()
docs = db.similarity_search(query)
query_time_1 = time.time() - start
#print(docs[0].page_content)

query = "What did the president say about Putin in the state of the union?"
print("===> Run Query: {} ".format(query))
start = time.time()
docs = db.similarity_search(query)
query_time_2 = time.time() - start
#print(docs[0].page_content)

query = "What did the president say assault weapon"
print("===> Run Query: {} ".format(query))
start = time.time()
docs = db.similarity_search(query)
query_time_3 = time.time() - start
#print(docs[0].page_content)

query = "What did the president say about Putin in the state of the union?"
print("===> Run Query: {} ".format(query))
start = time.time()
docs = db.similarity_search(query)
query_time_4 = time.time() - start
#print(docs[0].page_content)



print(f'document_load_time : {document_load_time}')
print(f'chunking_time      : {chunking_time}')
print(f'vector_store_time  : {vector_store_time}')
print(f'query_time_1       : {query_time_1}')
print(f'query_time_2       : {query_time_2}')
print(f'query_time_3       : {query_time_3}')
print(f'query_time_4       : {query_time_4}')
