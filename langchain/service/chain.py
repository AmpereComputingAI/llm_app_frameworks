import torch
torch.set_num_threads(32)

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.embeddings       import HuggingFaceEmbeddings
from langchain_community.vectorstores     import Chroma, FAISS
from langchain_community.llms             import LlamaCpp
from langchain_core.callbacks             import CallbackManager, StreamingStdOutCallbackHandler
from langchain.text_splitter              import CharacterTextSplitter
from langchain.chains.question_answering  import load_qa_chain
from langchain.chains                     import RetrievalQA

model_name="BAAI/bge-base-en-v1.5"
embedding_model = HuggingFaceEmbeddings(model_name=model_name, show_progress=False)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = DirectoryLoader( "./data/").load_and_split(text_splitter)

db = Chroma.from_documents(
                           persist_directory="./chroma_db",
                           documents=documents,
                           embedding=embedding_model)
'''
# to reuse above vector database
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
'''
#retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":4})
retriever = db.as_retriever()


model_path="model/llama-2-7b-chat.Q4_K_M.gguf"

llm = LlamaCpp(
    model_path=model_path,
    temperature=1,
    top_p=0.6,
    top_k = 35,
    callbacks=[StreamingStdOutCallbackHandler()],
    streaming=True,
    verbose=True,
)

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="refine",
    retriever=retriever,
    return_source_documents=False,
    verbose=True,
    )

