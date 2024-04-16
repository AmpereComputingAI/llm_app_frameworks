from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.embeddings       import HuggingFaceEmbeddings
from langchain_community.vectorstores     import Chroma
from langchain_community.llms             import LlamaCpp
from langchain_core.callbacks             import StreamingStdOutCallbackHandler
from langchain.text_splitter              import CharacterTextSplitter
from langchain.prompts                    import ChatPromptTemplate
from langchain.schema.runnable            import RunnablePassthrough
from langchain.schema.output_parser       import StrOutputParser

from fastapi import FastAPI, File, UploadFile
import uvicorn
import os

# prompt
template = """
Answer the user question based on your knowledge
Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# embeddings
model_name="BAAI/bge-small-en"
embedding_model = HuggingFaceEmbeddings(model_name=model_name, show_progress=False)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = DirectoryLoader( "./data/").load_and_split(text_splitter)
db = Chroma.from_documents(documents, embedding_model)
#retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})
retriever = db.as_retriever()

# model
# TODO replace StreamingStdOutCallbackHandler with Custom handler to stream answer
model_path="llama-2-7b-chat.Q4_K_M.gguf"
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.0,
    n_ctx=2048,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    verbose=True,
)

# chain
rag_chain = (
        {"context": retriever,
         "question": RunnablePassthrough()}
         | prompt
         | llm
         | StrOutputParser()
         )

# FastAPI app
app = FastAPI()


# update running to some good status and service at the startup of file
# default route
@app.get("/")
async def read_root():
    return { "LangChain RAG" : "Running" }

# file upload and vectordb update route
@app.post("/upload-file/")
async def create_upload_file(uploaded_file: UploadFile = File(...)):
    file_location = f"data/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())
    new_doc = TextLoader(file_location).load_and_split(text_splitter)
    db.add_documents(new_doc)
    os.remove(file_location)
    return {"info": f"file '{uploaded_file.filename}' moved into vectordb"}

# QA route
@app.post("/question/")
async def question_answer(question: str):
    response = rag_chain.invoke(question)
    return {"question:": f"question asked is {question} . answer {response}"}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)

