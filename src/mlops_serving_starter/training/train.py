"""Train XGBoost for aFRR capacity-price forecasting and log to MLflow.

Usage:
    python -m mlops_serving_starter.training.train --config configs/train_config.json --data data/sample.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from mlops_serving_starter.training.feature_engineering import prepare_features_for_horizon

logger = logging.getLogger(__name__)


def load_training_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_data(data_path: Path) -> pd.DataFrame:
    """Load CSV with timestamp_utc index (hourly time-series)."""
    df: pd.DataFrame = pd.read_csv(  # type: ignore[assignment]
        data_path, parse_dates=["timestamp_utc"], index_col="timestamp_utc"
    )
    df.sort_index(inplace=True)
    logger.info("Loaded %d rows from %s (%s → %s)", len(df), data_path, df.index.min(), df.index.max())
    return df


def train_and_log_model(
    config: dict,
    data_path: Path,
    tracking_uri: str | None = None,
    horizon: int = 1,
    hour: int | None = None,
) -> dict:
    """Train one XGBoost model for a given horizon, log everything to MLflow."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(config["experiment_name"])

    target = config["target"]
    df = load_data(data_path)

    # Feature engineering
    df_feat = prepare_features_for_horizon(df, target, horizon)

    if hour is not None:
        df_feat = df_feat[df_feat.index.hour == hour]
        logger.info("Filtered to hour %d: %d rows", hour, len(df_feat))

    # Chronological train/val split (last 20% for validation, like the production system)
    df_feat = df_feat.dropna(subset=[target])
    split_idx = int(len(df_feat) * (1 - config.get("test_size", 0.2)))
    train_data = df_feat.iloc[:split_idx]
    val_data = df_feat.iloc[split_idx:]

    X_train = train_data.drop(columns=[target])
    y_train = train_data[target]
    X_val = val_data.drop(columns=[target])
    y_val = val_data[target]

    logger.info("Train: %d rows, Val: %d rows, Features: %d", len(X_train), len(X_val), X_train.shape[1])

    run_name = f"{config.get('run_name', 'xgboost')}-h{horizon}"
    if hour is not None:
        run_name += f"-hr{hour:02d}"

    with mlflow.start_run(run_name=run_name) as run:
        model_params = config["model_params"].copy()
        random_state = model_params.pop("random_state", 42)

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
            early_stopping_rounds=20,
            **model_params,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Predictions and bias correction (like production)
        val_preds = model.predict(X_val)
        bias_correction = float((y_val.values - val_preds).mean())
        val_preds_corrected = np.clip(val_preds + bias_correction, a_min=0, a_max=None)

        metrics = {
            "mae": mean_absolute_error(y_val, val_preds_corrected),
            "rmse": float(np.sqrt(mean_squared_error(y_val, val_preds_corrected))),
            "bias_correction": bias_correction,
            "best_iteration": int(model.best_iteration),
        }

        mlflow.log_params(
            {
                "target": target,
                "horizon": horizon,
                "hour": hour if hour is not None else "all",
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                **{k: v for k, v in config["model_params"].items()},
            }
        )
        mlflow.log_metrics(metrics)

        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name=config.get("registered_model_name"),
        )

        return {
            "run_id": run.info.run_id,
            "model_uri": f"runs:/{run.info.run_id}/model",
            "metrics": metrics,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost time-series model with MLflow")
    parser.add_argument("--config", default="configs/train_config.json")
    parser.add_argument("--data", default="data/sample.csv", help="Path to hourly CSV data")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon in days (1-4)")
    parser.add_argument("--hour", type=int, default=None, help="Train model for a specific hour (0-23)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = train_and_log_model(
        config=load_training_config(Path(args.config)),
        data_path=Path(args.data),
        tracking_uri=args.tracking_uri,
        horizon=args.horizon,
        hour=args.hour,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
