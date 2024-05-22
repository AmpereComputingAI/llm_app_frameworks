For best results use Ampere Optimized local-rag Docker image on OCI A1 instance.

```
docker pull ghcr.io/amperecomputingai/local-rag:v0.0.1
# docker run -it --rm ghcr.io/amperecomputingai/local-rag:v0.0.1

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
 
