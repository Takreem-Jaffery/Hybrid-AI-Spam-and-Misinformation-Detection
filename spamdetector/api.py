"""FastAPI service.

Run:  uvicorn spamdetector.api:app --reload --port 8000

Endpoints
    POST /predict   spam score + explanation for one message
    POST /article/review   separate article veracity and headline clickbait classifications
    POST /advice    LLM security advice (needs GROQ_API_KEY on the server)
    GET  /health    which models are loaded
"""

from __future__ import annotations

import logging
import os
import warnings
from contextlib import asynccontextmanager
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import __version__, advisor
from .article_tasks import ArticleTaskModels
from .inference import Detector
from .scoring import DEFAULT_TFIDF_WEIGHT, Verdict, score_message

load_dotenv()
warnings.filterwarnings("ignore")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

MAX_MESSAGE_CHARS = 2000


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.detector = Detector.from_dir(use_transformer=os.getenv("DISABLE_TRANSFORMER", "0") != "1")
    app.state.article_tasks = ArticleTaskModels()
    yield


app = FastAPI(title="Message Risk Analyzer API", version=__version__, lifespan=lifespan)

# Restrict CORS to the dev UI by default (the original allowed every origin).
_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


class MessageIn(BaseModel):
    message: str = Field(max_length=MAX_MESSAGE_CHARS)
    is_forwarded: bool = False
    source: Literal["unknown", "friend", "group"] = "unknown"


class PredictOut(BaseModel):
    label: Literal["Spam", "Benign"]
    spam_score: float = Field(description="0-100 spam score. Not a calibrated probability.")
    risk: Literal["Low", "Medium", "High"]
    riskColor: str
    model_used: str
    explanation: str
    reasons: list[str]


class AdviceOut(BaseModel):
    verdict: str
    reasoning: str
    actions: list[str]
    report_to: str | None


class ArticleReviewIn(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    body: str = Field(min_length=1, max_length=20000)


class ArticleReviewOut(BaseModel):
    veracity_category: str
    veracity_model_score: float
    veracity_scores: dict[str, float]
    clickbait_category: str
    clickbait_model_score: float
    clickbait_scores: dict[str, float]
    evaluation: dict[str, dict[str, float | str]]
    model_used: str


def _analyse(detector: Detector, body: MessageIn) -> Verdict:
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    weight = float(os.getenv("TFIDF_WEIGHT", DEFAULT_TFIDF_WEIGHT))
    return score_message(
        body.message,
        is_forwarded=body.is_forwarded,
        source=body.source,
        tfidf_score=detector.tfidf_score(body.message),
        transformer_score=detector.transformer_score(body.message),
        tfidf_label=detector.tfidf.label if detector.tfidf else "TF-IDF classifier",
        tfidf_weight=weight,
    )


def _to_out(v: Verdict) -> PredictOut:
    return PredictOut(
        label=v.label, spam_score=v.spam_score, risk=v.risk, riskColor=v.risk_color,
        model_used=v.model_used, explanation=v.explanation, reasons=v.reasons,
    )


@app.post("/predict", response_model=PredictOut)
def predict(body: MessageIn, request: Request) -> PredictOut:
    return _to_out(_analyse(request.app.state.detector, body))


@app.post("/article/review", response_model=ArticleReviewOut)
def review_article(body: ArticleReviewIn, request: Request) -> ArticleReviewOut:
    if not body.title.strip() or not body.body.strip():
        raise HTTPException(status_code=400, detail="Article title and body cannot be empty.")
    try:
        result = request.app.state.article_tasks.review(body.title.strip(), body.body.strip())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ArticleReviewOut(**result)


@app.post("/advice", response_model=AdviceOut)
async def advice(body: MessageIn, request: Request) -> AdviceOut:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="The AI advisor is not configured (set GROQ_API_KEY on the server).")
    # The verdict is recomputed server-side instead of trusting one sent by the browser.
    verdict = await run_in_threadpool(_analyse, request.app.state.detector, body)
    try:
        result = await advisor.request_advice(
            body.message, verdict, api_key=api_key, model=os.getenv("GROQ_MODEL", advisor.DEFAULT_MODEL)
        )
    except advisor.AdvisorError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return AdviceOut(**result.model_dump())


@app.get("/health")
def health(request: Request) -> dict:
    return {
        **request.app.state.detector.status(),
        **request.app.state.article_tasks.status(),
        "version": __version__,
    }
