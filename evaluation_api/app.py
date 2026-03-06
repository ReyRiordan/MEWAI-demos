"""
Demo Evaluation API — no auth, no database.

POST /evaluate accepts a rubric + transcript + responses and returns
the full Evaluation structure.
"""

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from evaluation import evaluate
from models import Evaluation, EvaluateRequest

app = FastAPI(title="MEWAI Evaluation API (Demo)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/evaluate", response_model=Evaluation)
async def evaluate_endpoint(request: EvaluateRequest) -> Evaluation:
    return await evaluate(request.rubric, request.transcript, request.responses)
