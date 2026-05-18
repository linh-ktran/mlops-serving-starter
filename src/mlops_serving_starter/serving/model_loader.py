from __future__ import annotations

import importlib

import numpy as np
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
        # Cast columns to match the MLflow model signature types
        if self.model.metadata and self.model.metadata.signature:
            input_schema = self.model.metadata.signature.inputs
            for col_spec in input_schema.inputs:
                col_name = col_spec.name
                if col_name not in frame.columns:
                    continue
                if col_spec.type.name == "double":
                    frame[col_name] = frame[col_name].astype(np.float64)
                elif col_spec.type.name == "long":
                    frame[col_name] = frame[col_name].astype(np.int64)
        predictions = self.model.predict(frame)
        return [float(value) for value in predictions]

