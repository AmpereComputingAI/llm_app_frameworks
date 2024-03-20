This folder contains sample applications to run LLM based application using LangChain framework along with VectorDB.
LangChain simplify creation of applications based on large language models. VectorDB help to have long term memory.

Following are test applications using different VectorDB.
```
1. chromadb_example1.py
2. faiss_example1.py
3. weaviate_example1.py
```

Document used “state_of_the_union.txt”. It must be present in the same folder as test application or update the test
application as per location of the document.  Other text document can be used in place of “state_of_the_union.txt”

Following packages are required to run test applications:
```
# pip install langchain
# pip install langchain_community
# pip install sentence-transformers
# pip install weaviate-client
# pip install chromadb
# pip install faiss-cpu
```

To run test application, execute following command:

# python  test_application.py

Test application using Chroma as VectorDB
```
# python chromadb_example1.py
```
Test application using Faiss as VectorDB
```
# python faiss_example1.py
```
Test application using Weaviate as VectorDB
```
# python weaviate_example1.py
```

Notre:  “weaviate_example1.py” will look for VectorDB on local host. Make sure to start Weaviate VectorDB server on locahost.

