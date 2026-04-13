from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def load_training_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def train_and_log_model(config: dict, tracking_uri: str | None = None) -> dict:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(config["experiment_name"])
    dataset = load_iris(as_frame=True)

    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=config.get("test_size", 0.2),
        random_state=config["model_params"]["random_state"],
        stratify=dataset.target,
    )

    with mlflow.start_run(run_name=config.get("run_name", "baseline-rf")) as run:
        model = RandomForestClassifier(**config["model_params"])
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "f1_weighted": f1_score(y_test, predictions, average="weighted"),
        }

        mlflow.log_params(config["model_params"])
        mlflow.log_metrics(metrics)

        signature = mlflow.models.infer_signature(x_train, model.predict(x_train))
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
    parser = argparse.ArgumentParser(description="Train and log a model with MLflow")
    parser.add_argument("--config", default="configs/train_config.json")
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()

    result = train_and_log_model(
        config=load_training_config(Path(args.config)),
        tracking_uri=args.tracking_uri,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

