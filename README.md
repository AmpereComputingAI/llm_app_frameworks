# LLM Application Frameworks 
Contains scripts that integrates LLM and Vector DB to test Retrieval Augmented Generation use cases. Two commonly used open-source frameworks that can be potential candidates to integrate with llama-cpp/llama-cpp-python:

[**LangChain**](https://python.langchain.com/docs/integrations/text_embedding/llamacpp)
and
[**LlamaIndex**](https://docs.llamaindex.ai/en/stable/examples/llm/llama_2_llama_cpp.html#llamacpp)

## Benchmarking
```
cd langchain # cd llamaindex
pip3 install -r requirements.txt
./bench.sh
```