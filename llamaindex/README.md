Install python packages:

```
pip install --upgrade pip
pip install llama-index-readers-file pymupdf
pip install llama-index-embeddings-huggingface
pip install llama-index-llms-llama-cpp
pip install llama-cpp-python
pip install psycopg2-binary pgvector asyncpg "sqlalchemy[asyncio]" greenlet
pip install llama-index-vector-stores-chroma
pip install llama-index chromadb 
pip install sentence-transformers
pip install pydantic==1.10.11
```

Get llm_app_frameworks.
```
git clone git@github.com:AmpereComputingAI/llm_app_frameworks.git
cd llm_app_frameworks/llamaindex
```

Start RAG application
```
python llamaindex-cli.py
```

In QA command line ask questions.

Sample Reference session

```
Welcome to llama CLI. Reserved first words help, upload and quit
               Type "help" for available commands.
llama QA >> who is president of USA?
 president of the United States is Joe Biden.
llama QA >> upload news_usa.txt
Uploading  news_usa.txt [Done]
llama QA >> who is president of USA?
 answer to the query is "Tom Cruise".
```

Fake news is uploaded to RAG. Which changes the answer from "Joe Biden" to "Tom Cruise" 

