import os
os.environ["AIO_NUM_THREADS"]="32"
import torch
torch.set_num_threads(32)
import cmd

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.embeddings       import HuggingFaceEmbeddings
from langchain_community.vectorstores     import Chroma
from langchain_community.llms             import LlamaCpp
from langchain_core.callbacks             import StreamingStdOutCallbackHandler
from langchain.text_splitter              import CharacterTextSplitter
from langchain.prompts                    import ChatPromptTemplate
from langchain.schema.runnable            import RunnablePassthrough
from langchain.schema.output_parser       import StrOutputParser

from langchain.callbacks.base             import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Any, Dict, List

class CustomAsyncHandler(AsyncCallbackHandler):
    thinking=False
    """Async callback handler that can be used to handle callbacks from langchain."""
    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""
        formatted_prompts = "\n".join(prompts)
        # extract question asked
        print(formatted_prompts.split("Context:")[0].split("Human:")[1])
        print("Answer: ", end="")

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when chain ends running."""
        print()

template = """
Question: {question}
Context: {context}
Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

model_name="BAAI/bge-small-en"
embedding_model = HuggingFaceEmbeddings(model_name=model_name, show_progress=False)
#embedding_model.client.forward = torch.compile(embedding_model.client.forward)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = DirectoryLoader( "./data/").load_and_split(text_splitter)
db = Chroma.from_documents(documents, embedding_model)
#retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})
retriever = db.as_retriever()

model_path="llama-2-7b-chat.Q4_K_M.gguf"

llm = LlamaCpp(
    model_path=model_path,
    temperature=0.0,
    n_ctx=2048,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler(), CustomAsyncHandler()],
    verbose=False,
)

rag_chain = (
        {"context": retriever,
         "question": RunnablePassthrough()}
         | prompt
         | llm
         | StrOutputParser()
         )

class LlmCLI(cmd.Cmd):
    prompt = '>> '
    intro = 'Welcome to LlmCLI. Type "help" for available commands.'

    def do_question(self, line):
        """Ask question."""
        # invoke chain with question asked
        rag_chain.invoke(line)

    def do_quit(self, line):
        """Exit the CLI."""
        return True
        
    def do_upload(self, line):
        """Upload text file into vectordb"""
        if os.path.isfile(line):
            print("Uploading ", line, end="")
            new_doc = TextLoader(line).load_and_split(text_splitter)
            ids=db.add_documents(new_doc)
            print(" [Done]")
        else:
            print("File ", line,  "not found")

if __name__ == '__main__':
    LlmCLI().cmdloop()

