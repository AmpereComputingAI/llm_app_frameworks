For best results use Ampere Optimized local-rag Docker image on OCI A1 instance.

```
docker pull ghcr.io/amperecomputingai/local-rag:v0.0.1
# docker run -it --rm ghcr.io/amperecomputingai/local-rag:v0.0.1

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

