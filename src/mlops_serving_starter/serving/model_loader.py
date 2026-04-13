from __future__ import annotations

import importlib

import pandas as pd


class ModelService:
    def __init__(self, model_uri: str, tracking_uri: str | None = None) -> None:
        mlflow = importlib.import_module("mlflow")
        if tracking_uri:

            mlflow.set_tracking_uri(tracking_uri)
        self.model_uri = model_uri
        self.model = mlflow.pyfunc.load_model(model_uri)

    def predict(self, records: list[dict]) -> list[float]:
        frame = pd.DataFrame(records)
        predictions = self.model.predict(frame)
        return [float(value) for value in predictions]

