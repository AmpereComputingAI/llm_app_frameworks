Install python packages:

```
pip install --upgrade pip
pip install langchain
pip install langchain_community
pip install sentence-transformers
pip install transformers
pip install sentence-transformers
pip install llama-cpp-python
pip install unstructured
pip install pydantic==1.10.13
pip install sse_starlette
pip install chromadb
```
Get llm_app_frameworks.
```
git clone git@github.com:AmpereComputingAI/llm_app_frameworks.git
cd llm_app_frameworks/llamaindex
``` 

Start RAG application
```
python langchain-cli.py
```

In QA command line ask questions.

Sample Reference session

```
Welcome to llama CLI. Reserved first words help, upload and quit
               Type "help" for available commands.
llama QA >> Who is president of India?
The current President of India is Ram Nath Kovind.
llama QA >> upload news_india.txt
Uploading  news_india.txt [Done]
llama QA >> Who is president of India?
Droupadi Murmu
llama QA >> quit
```

At the time of "llama-2-7b-chat.Q4_K_M.gguf" training, president of India was "Ram Nath Kovind"
With "upload news_india.txt", latest news regarging president of India is uploaded to RAG.
Asking same question again provides the correct answer "Droupadi Murmu"
 
