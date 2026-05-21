from __future__ import annotations

import os
from functools import lru_cache

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mlops_serving_starter.serving.model_loader import ModelService


class PredictionRequest(BaseModel):
    """Each record is a pre-computed feature vector (lags, rolling stats, forecasts, calendar flags)."""

    records: list[dict] = Field(min_length=1)


class ForecastResponse(BaseModel):
    predictions: list[float]
    target: str | None = None
    horizon: int | None = None
    unit: str = "EUR/MW"


@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    model_uri = os.getenv("MODEL_URI")
    if not model_uri:
        raise RuntimeError("MODEL_URI env var is required (e.g. runs:/<RUN_ID>/model)")
    return ModelService(model_uri=model_uri, tracking_uri=os.getenv("MLFLOW_TRACKING_URI"))


def create_app() -> FastAPI:
    app = FastAPI(
        title="aFRR Capacity Price Forecast",
        version="0.2.0",
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/predict", response_model=ForecastResponse)
    def predict(payload: PredictionRequest) -> ForecastResponse:
        try:
            model_service = get_model_service()
            preds = model_service.predict(payload.records)
            preds = [max(0.0, p) for p in preds]
            return ForecastResponse(
                predictions=preds,
                target=os.getenv("TARGET_NAME", "afrr_capacity_price_up"),
                horizon=int(os.getenv("FORECAST_HORIZON", "1")),
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()


def run() -> None:
    uvicorn.run("mlops_serving_starter.api.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
