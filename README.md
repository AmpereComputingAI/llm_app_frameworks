# LLM Application Frameworks 
Contains scripts that integrates LLM and Vector DB (ChromaDB) to test Retrieval Augmented Generation use cases. Two commonly used open-source frameworks that can be potential candidates to integrate with llama-cpp/llama-cpp-python:

CLI based RAG application for 
[**LangChain**](https://github.com/AmpereComputingAI/llm_app_frameworks/tree/master/langchain) and 
[**LlamaIndex**](https://github.com/AmpereComputingAI/llm_app_frameworks/tree/master/llamaindex)

Preduild Docker image (Image is based on Ampere Optimized PyTorch and llama-cpp):

```
# docker pull ghcr.io/amperecomputingai/local-rag:v0.0.1
# docker run -it --rm ghcr.io/amperecomputingai/local-rag:v0.0.1
```

First step is to get the llama model
```
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
```

Running LangChain RAG Application
```
# python langchain-cli.py

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

At the time of "llama-2-7b-chat.Q4_K_M.gguf" training, president of India was "Ram Nath Kovind" With "upload news_india.txt", latest news regarging president of India is uploaded to RAG. Asking same question again provides the correct answer "Droupadi Murmu"

Running LlamaIndex RAG Application
```
# python llamaindex-cli.py

Welcome to llama CLI. Reserved first words help, upload and quit
               Type "help" for available commands.
llama QA >> who is president of USA?
 president of the United States is Joe Biden.
llama QA >> upload news_usa.txt
Uploading  news_usa.txt [Done]
llama QA >> who is president of USA?
 answer to the query is "Tom Cruise".
llama QA >> quit
```

Fake news (news_usa.txt) is uploaded to RAG. Which changes the answer from "Joe Biden" to "Tom Cruise"

For best results use Ampere Optimized local-rag Docker image on **OCI A1 instance**.

System requirement: 64 OCPUs + 128GB Memory.
