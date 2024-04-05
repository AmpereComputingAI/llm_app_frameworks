from fastapi import FastAPI
from langserve import add_routes
from chain import chain

app = FastAPI()

add_routes(app, chain, path="/RAG")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
