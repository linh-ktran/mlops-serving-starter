from __future__ import annotations

import importlib

import numpy as np
import pandas as pd


class ModelService:
    """Thin wrapper around an MLflow pyfunc model."""

    def __init__(self, model_uri: str, tracking_uri: str | None = None) -> None:
        mlflow = importlib.import_module("mlflow")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self.model = mlflow.pyfunc.load_model(model_uri)

    def predict(self, records: list[dict]) -> list[float]:
        df = pd.DataFrame(records)
        self._cast_types(df)
        return [float(v) for v in self.model.predict(df)]

    def _cast_types(self, df: pd.DataFrame) -> None:
        """Align column dtypes to the MLflow model signature."""
        if not (self.model.metadata and self.model.metadata.signature):
            return
        for col in self.model.metadata.signature.inputs.inputs:
            if col.name not in df.columns:
                continue
            if col.type.name == "double":
                df[col.name] = df[col.name].astype(np.float64)
            elif col.type.name == "long":
                df[col.name] = df[col.name].astype(np.int64)
