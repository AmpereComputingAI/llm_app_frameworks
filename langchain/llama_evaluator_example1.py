from langchain_community.embeddings import LlamaCppEmbeddings
from langchain.evaluation import load_evaluator
#embedding_model = LlamaCppEmbeddings(model_path="/onspecta/sandbox/llama.cpp/models/7B/llama-2-7b.Q4_K_M.gguf")
embedding_model = LlamaCppEmbeddings(model_path="./models/embedding_models/bge-base-en-v1.5-ggml-model-fp16.gguf")

llama_evaluator = load_evaluator("embedding_distance", embeddings=embedding_model)

prediction = "I shall go"
reference = "I shan't go"
print("Prediction String: {}".format(prediction))
print("Reference String: {}".format(reference))
evaluation = llama_evaluator.evaluate_strings(prediction = prediction, reference = reference)
print("Distance Evaluation: {}".format(evaluation))

prediction = "I shall go"
reference = "I will go"
print("Prediction String: {}".format(prediction))
print("Reference String: {}".format(reference))
evaluation = llama_evaluator.evaluate_strings(prediction = prediction, reference = reference)
print("Distance Evaluation: {}".format(evaluation))

prediction = "The King and Queen"
reference = "The Prince and Princess"
print("Prediction String: {}".format(prediction))
print("Reference String: {}".format(reference))
evaluation = llama_evaluator.evaluate_strings(prediction = prediction, reference = reference)
print("Distance Evaluation: {}".format(evaluation))

prediction = "The King and Queen"
reference = "The Apple and Orange"
print("Prediction String: {}".format(prediction))
print("Reference String: {}".format(reference))
evaluation = llama_evaluator.evaluate_strings(prediction = prediction, reference = reference)
print("Distance Evaluation {}".format(evaluation))
