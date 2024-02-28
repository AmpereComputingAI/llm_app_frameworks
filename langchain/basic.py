from langchain_community.embeddings import LlamaCppEmbeddings

print("Loading Embedding Model...")
llama = LlamaCppEmbeddings(model_path="./models/bge-base-en-v1.5-ggml-model-fp16.gguf")
#llama = LlamaCppEmbeddings(model_path="./models/7B/llama-2-7b.Q4_K_M.gguf")
text = "This is a test document."
print("Start Query: {}".format(text))
text_embedding = llama.embed_query(text)
print("Text embedding size {}".format(len(text_embedding)))
for i in range(len(text_embedding)):
  print(text_embedding[i])
