from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from .demo import DEMO_HTML
from .service import RecommendationService


app = FastAPI(
    title="Amazon Video Games Recommender",
    version="3.0.0",
    description="Hybrid Top-K recommendations with metadata-aware discovery.",
)
service = RecommendationService(os.getenv("MODEL_PATH", "artifacts/bpr_model.pt"))


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def demo() -> str:
    return DEMO_HTML


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/recommend/{user_id}")
def recommend(user_id: str, k: int = Query(default=10, ge=1, le=100)) -> dict:
    try:
        return service.recommend(user_id, k)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/similar/{item_id}")
def similar(item_id: str, k: int = Query(default=10, ge=1, le=100)) -> dict:
    try:
        return service.similar(item_id, k)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Unknown item") from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
