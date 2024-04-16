from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.embeddings       import HuggingFaceEmbeddings
from langchain_community.vectorstores     import Chroma
from langchain_community.llms             import LlamaCpp
from langchain_core.callbacks             import StreamingStdOutCallbackHandler
from langchain.text_splitter              import CharacterTextSplitter
from langchain.prompts                    import ChatPromptTemplate
from langchain.schema.runnable            import RunnablePassthrough
from langchain.schema.output_parser       import StrOutputParser

template = """
Question: {question}
Context: {context}
Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

model_name="BAAI/bge-small-en"
embedding_model = HuggingFaceEmbeddings(model_name=model_name, show_progress=False)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = DirectoryLoader( "./data/").load_and_split(text_splitter)
db = Chroma.from_documents(documents, embedding_model)
retriever = db.as_retriever()

model_path="llama-2-7b-chat.Q4_K_M.gguf"

llm = LlamaCpp(
    model_path=model_path,
    temperature=0.0,
    n_ctx=1024,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    verbose=True,
)

rag_chain = (
        {"context": retriever,
         "question": RunnablePassthrough()}
         | prompt
         | llm
         | StrOutputParser()
         )
rag_chain.invoke("Who is president of USA?")

new_test_doc="news_usa.txt"
new_doc = TextLoader(new_test_doc).load_and_split(text_splitter)
ids=db.add_documents(new_doc)

rag_chain.invoke("Who is president of USA?")
rag_chain.invoke("Who is Joe Biden?")
rag_chain.invoke("Who is Tom Cruise?")
rag_chain.invoke("Who is 47th President of the United States?")

