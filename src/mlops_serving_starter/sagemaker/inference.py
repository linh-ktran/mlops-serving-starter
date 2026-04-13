from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import mlflow.pyfunc
import pandas as pd


DEFAULT_MODEL_DIR_NAME = "model"

# SageMaker calls these four functions in order for every inference request.
# model_fn is called once at container start, the others on each request.
# This matches the DefaultInferenceHandler interface from the sagemaker-inference SDK.


def model_fn(model_dir: str):
    model_path = Path(model_dir) / DEFAULT_MODEL_DIR_NAME
    return mlflow.pyfunc.load_model(str(model_path))


def input_fn(request_body: str, request_content_type: str):
    if request_content_type == "application/json":
        payload = json.loads(request_body)
        return pd.DataFrame(payload["records"])

    if request_content_type == "text/csv":
        return pd.read_csv(StringIO(request_body))

    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data: pd.DataFrame, model):
    return model.predict(input_data)


def output_fn(prediction, accept: str):
    if accept == "application/json":
        return json.dumps({"predictions": [float(item) for item in prediction]})

    if accept == "text/csv":
        frame = pd.DataFrame({"prediction": prediction})
        return frame.to_csv(index=False)

    raise ValueError(f"Unsupported accept type: {accept}")

