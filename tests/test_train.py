from pathlib import Path

from mlops_serving_starter.training.train import train_and_log_model


def test_train_and_log_model(tmp_path):
    tracking_uri = f"file://{tmp_path / 'mlruns'}"
    data_path = Path("data/sample.csv")

    config = {
        "experiment_name": "test-exp",
        "run_name": "test-run",
        "registered_model_name": None,
        "target": "afrr_capacity_price_up",
        "test_size": 0.2,
        "model_params": {
            "n_estimators": 10,
            "max_depth": 3,
            "random_state": 42,
        },
    }

    result = train_and_log_model(
        config=config, data_path=data_path, tracking_uri=tracking_uri, horizon=1
    )

    assert result["run_id"]
    assert result["model_uri"].startswith("runs:/")
    assert result["metrics"]["mae"] > 0

