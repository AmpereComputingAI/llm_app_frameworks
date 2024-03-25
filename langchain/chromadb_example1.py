import torch
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
import time

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5", show_progress=True)

#embedding_model.client.forward = torch.compile(embedding_model.client.forward, backend='aio', options={"modelname": "BAAI/bge-large-en-v1.5"})
embedding_model.client.forward = torch.compile(embedding_model.client.forward)

print("===> Loading Documents ....")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
raw_documents = TextLoader('./data/state_of_the_union.txt').load_and_split(text_splitter)
raw_documents += TextLoader('./data/paul_graham_essay.txt').load_and_split(text_splitter) 
print("===> Chunking Documents ....")
documents = text_splitter.split_documents(raw_documents)
print("===> Vector store ....")
start = time.time()

db = Chroma.from_documents(documents, embedding_model)

end = time.time()

print("Embedding generation took", end - start)

exit(0)

query = "What did the president say assault weapon"
print("===> Run Query: {} ".format(query))
docs = db.similarity_search(query)
print(docs[0].page_content)
