import os
model_url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
if False == os.path.isfile('llama-2-7b-chat.Q4_K_M.gguf'):
    print('Please download model llama-2-7b-chat.Q4_K_M.gguf')
    print(' from: ', model_url)
    quit()

import torch
torch.set_num_threads(32)
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.embeddings       import HuggingFaceEmbeddings
from langchain_community.vectorstores     import Chroma
from langchain_community.llms             import LlamaCpp
from langchain_core.callbacks             import CallbackManager, StreamingStdOutCallbackHandler
from langchain.text_splitter              import CharacterTextSplitter
from langchain.chains.question_answering  import load_qa_chain
from langchain.chains                     import RetrievalQA

model_name="BAAI/bge-small-en-v1.5"
embedding_model = HuggingFaceEmbeddings(model_name=model_name, show_progress=False)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = DirectoryLoader( "./data/").load_and_split(text_splitter)
db = Chroma.from_documents(documents, embedding_model)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

model_path="llama-2-7b-chat.Q4_K_M.gguf"

llm = LlamaCpp(
    model_path=model_path,
    temperature=0.1,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=False,
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="refine",
    retriever=retriever,
    return_source_documents=True)

question = "Question: What did the president say about Putin in the state of the union?"
print('\nQuestion : ', question)
print('Answer   :', end='')
qa_chain.invoke(question)

question = "What did the author do growing up?"
print('\nQuestion : ', question)
print('Answer   :', end='')
qa_chain.invoke(question)
print()

