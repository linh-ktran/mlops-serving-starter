"""SageMaker serving container — implements /ping and /invocations."""

from __future__ import annotations

import json
import logging
import os
import traceback
from contextlib import asynccontextmanager
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Response

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

_model = None


def load_model():
    """Load the MLflow model from the directory SageMaker unpacks model.tar.gz into."""
    global _model
    if _model is not None:
        return _model

    import mlflow.pyfunc

    model_dir = Path(os.environ.get("MODEL_DIR", "/opt/ml/model"))
    logger.info("Loading model from %s", model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"MODEL_DIR does not exist: {model_dir}")

    # model.tar.gz is packaged with a nested 'model/' directory
    model_path = model_dir / "model" if (model_dir / "model").exists() else model_dir

    if not (model_path / "MLmodel").exists():
        contents = [str(p) for p in model_dir.rglob("*")][:30]
        raise FileNotFoundError(f"No MLmodel file at {model_path}. Contents: {contents}")

    _model = mlflow.pyfunc.load_model(str(model_path))
    logger.info("Model loaded from %s", model_path)
    return _model


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Pre-load model at startup so /ping responds immediately."""
    try:
        load_model()
    except Exception as exc:
        logger.error("Failed to load model at startup: %s", exc)
        logger.error(traceback.format_exc())
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/ping")
def ping():
    """Health check — SageMaker polls this until it returns 200."""
    try:
        load_model()
        return Response(status_code=200, content="OK")
    except Exception as exc:
        logger.error("Ping failed: %s", exc)
        return Response(status_code=503, content=str(exc))


@app.post("/invocations")
async def invocations(request: Request):
    """Prediction endpoint — receives feature records, returns forecasts."""
    content_type = request.headers.get("content-type", "")
    body = await request.body()

    if "application/json" in content_type:
        payload = json.loads(body)
        df = pd.DataFrame(payload["records"])
    elif "text/csv" in content_type:
        df = pd.read_csv(StringIO(body.decode("utf-8")))
    else:
        return Response(status_code=415, content=f"Unsupported content type: {content_type}")

    model = load_model()

    # Match column types to what the model expects
    if model.metadata and model.metadata.signature:
        for col_spec in model.metadata.signature.inputs.inputs:
            if col_spec.name not in df.columns:
                continue
            if col_spec.type.name == "double":
                df[col_spec.name] = df[col_spec.name].astype(np.float64)
            elif col_spec.type.name == "long":
                df[col_spec.name] = df[col_spec.name].astype(np.int64)

    predictions = model.predict(df)
    predictions = [max(0.0, float(p)) for p in predictions]

    return Response(
        status_code=200,
        content=json.dumps({"predictions": predictions}),
        media_type="application/json",
    )
