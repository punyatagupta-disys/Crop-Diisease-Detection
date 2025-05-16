from fastapi import FastAPI, Request
from pydantic import BaseModel
from ragmodel import get_rag_answer  # Your custom RAG logic

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/rag")
async def rag_query(data: Query):
    answer = get_rag_answer(data.question)
    return {"answer": answer}

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
