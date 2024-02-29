from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

print("===> Loading Documents ....")
raw_documents = TextLoader('./state_of_the_union.txt').load()
print("===> Chunking Documents ....")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
print("===> Vector store ....")
db = Chroma.from_documents(documents, embeddings)

query = "What did the president say assault weapon"
print("===> Run Query: {} ".format(query))
docs = db.similarity_search(query)
print(docs[0].page_content)
