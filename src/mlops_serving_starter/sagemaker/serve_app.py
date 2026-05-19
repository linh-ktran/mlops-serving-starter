"""FastAPI app for SageMaker serving (implements /ping and /invocations).

SageMaker requires:
- GET /ping → 200 (health check)
- POST /invocations → predictions
"""

from __future__ import annotations

import logging
import os
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Global model reference (loaded once)
_model = None


def load_model():
    """Load the MLflow model from /opt/ml/model."""
    global _model
    if _model is None:
        import mlflow.pyfunc
        model_dir = os.environ.get("MODEL_DIR", "/opt/ml/model")
        logger.info(f"Loading model from MODEL_DIR={model_dir}")

        # Log directory contents for debugging
        model_dir_path = Path(model_dir)
        if model_dir_path.exists():
            contents = list(model_dir_path.rglob("*"))
            logger.info(f"Contents of {model_dir}: {[str(c) for c in contents[:50]]}")
        else:
            logger.error(f"MODEL_DIR does not exist: {model_dir}")
            raise FileNotFoundError(f"MODEL_DIR does not exist: {model_dir}")

        # Try /opt/ml/model/model first (packaged with nested 'model' dir)
        model_path = model_dir_path / "model"
        if not model_path.exists():
            logger.info(f"{model_path} not found, falling back to {model_dir_path}")
            model_path = model_dir_path

        # Check for MLmodel file (required by MLflow)
        mlmodel_file = model_path / "MLmodel"
        if not mlmodel_file.exists():
            logger.error(
                f"MLmodel file not found at {mlmodel_file}. "
                f"The model artifact may be packaged incorrectly. "
                f"Ensure model.tar.gz contains a valid MLflow model directory."
            )
            raise FileNotFoundError(f"MLmodel not found at {mlmodel_file}")

        logger.info(f"Loading MLflow model from: {model_path}")
        _model = mlflow.pyfunc.load_model(str(model_path))
        logger.info("Model loaded successfully")
    return _model


@app.on_event("startup")
def startup_event():
    """Pre-load model at container startup for faster health checks."""
    logger.info("SageMaker serving container starting up...")
    try:
        load_model()
        logger.info("Model pre-loaded successfully at startup")
    except Exception as e:
        logger.error(f"Failed to pre-load model at startup: {e}")
        logger.error(traceback.format_exc())


@app.get("/ping")
def ping():
    """Health check — SageMaker calls this every few seconds."""
    try:
        load_model()
        return Response(status_code=200, content="OK")
    except Exception as e:
        logger.error(f"Ping health check FAILED: {e}")
        logger.error(traceback.format_exc())
        return Response(status_code=503, content=f"Model not loaded: {e}")


@app.post("/invocations")
async def invocations(request: Request):
    """Prediction endpoint — called by SageMaker runtime."""
    content_type = request.headers.get("content-type", "")
    body = await request.body()

    if "application/json" in content_type:
        import json
        payload = json.loads(body)
        df = pd.DataFrame(payload["records"])
    elif "text/csv" in content_type:
        from io import StringIO
        df = pd.read_csv(StringIO(body.decode("utf-8")))
    else:
        return Response(status_code=415, content=f"Unsupported content type: {content_type}")

    # Cast types to match model signature
    model = load_model()
    if model.metadata and model.metadata.signature:
        for col_spec in model.metadata.signature.inputs.inputs:
            col_name = col_spec.name
            if col_name not in df.columns:
                continue
            if col_spec.type.name == "double":
                df[col_name] = df[col_name].astype(np.float64)
            elif col_spec.type.name == "long":
                df[col_name] = df[col_name].astype(np.int64)

    predictions = model.predict(df)
    predictions = [max(0.0, float(p)) for p in predictions]

    import json
    return Response(
        status_code=200,
        content=json.dumps({"predictions": predictions}),
        media_type="application/json",
    )


