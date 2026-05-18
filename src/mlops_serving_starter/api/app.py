from __future__ import annotations

import os
from functools import lru_cache

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mlops_serving_starter.serving.model_loader import ModelService


class PredictionRequest(BaseModel):
    """Request body for the /predict endpoint.

    For time-series forecasting, each record should contain the engineered
    features for a single timestamp (lag values, rolling stats, forecast
    features, calendar flags).  The feature set must match the model's
    training signature.

    Example record keys for aFRR:
        afrr_capacity_price_up_lag_1_h1, rolling_mean_7, gas_price_forecast,
        holiday_status, weekend_status, ...
    """
    records: list[dict] = Field(min_length=1)


class ForecastResponse(BaseModel):
    """Response body with predictions and optional metadata."""
    predictions: list[float]
    target: str | None = None
    horizon: int | None = None
    unit: str = "EUR/MW"


@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    model_uri = os.getenv("MODEL_URI")
    if not model_uri:
        # set MODEL_URI=runs:/<RUN_ID>/model before starting the server
        raise RuntimeError("MODEL_URI must be set, for example runs:/<RUN_ID>/model")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    return ModelService(model_uri=model_uri, tracking_uri=tracking_uri)


def create_app() -> FastAPI:
    app = FastAPI(
        title="aFRR Capacity Price Forecast API",
        version="0.2.0",
        description="Serves XGBoost time-series forecasts for French aFRR capacity prices.",
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/predict", response_model=ForecastResponse)
    def predict(payload: PredictionRequest) -> ForecastResponse:
        """Generate forecasts from pre-computed feature records.

        The client is responsible for computing the feature vector
        (lags, rolling stats, exogenous forecasts, calendar flags) and
        sending it as a list of dicts — one per timestamp to forecast.
        """
        try:
            model_service = get_model_service()
            predictions = model_service.predict(payload.records)
            # Capacity prices cannot be negative
            predictions = [max(0.0, p) for p in predictions]
            return ForecastResponse(
                predictions=predictions,
                target=os.getenv("TARGET_NAME", "afrr_capacity_price_up"),
                horizon=int(os.getenv("FORECAST_HORIZON", "1")),
            )
        except Exception as exc:  # pragma: no cover - integration runtime safety
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()


def run() -> None:
    uvicorn.run("mlops_serving_starter.api.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()

