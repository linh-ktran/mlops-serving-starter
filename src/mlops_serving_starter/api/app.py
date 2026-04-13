from __future__ import annotations

import os
from functools import lru_cache

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mlops_serving_starter.serving.model_loader import ModelService


class PredictionRequest(BaseModel):
    records: list[dict] = Field(min_length=1)


class PredictionResponse(BaseModel):
    predictions: list[float]


@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    model_uri = os.getenv("MODEL_URI")
    if not model_uri:
        raise RuntimeError("MODEL_URI must be set, for example runs:/<RUN_ID>/model")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    return ModelService(model_uri=model_uri, tracking_uri=tracking_uri)


def create_app() -> FastAPI:
    app = FastAPI(title="MLOps Serving Starter", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest) -> PredictionResponse:
        try:
            model_service = get_model_service()
            predictions = model_service.predict(payload.records)
            return PredictionResponse(predictions=predictions)
        except Exception as exc:  # pragma: no cover - integration runtime safety
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()


def run() -> None:
    uvicorn.run("mlops_serving_starter.api.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()

