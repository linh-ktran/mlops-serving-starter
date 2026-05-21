from pathlib import Path

import numpy as np
import pandas as pd

from mlops_serving_starter.training.train import train_and_log_model


def _generate_test_data(path: Path) -> None:
    """Generate a small synthetic dataset for testing."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2025-10-01", periods=60 * 24, freq="h", tz="UTC", name="timestamp_utc")
    n = len(idx)
    hour = idx.hour

    df = pd.DataFrame(
        {
            "afrr_capacity_price_up": 20 + 10 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 5, n),
            "fcr_price_symmetric": 15 + rng.normal(0, 2, n),
            "consumption_forecast": 50000 + rng.normal(0, 2000, n),
            "gas_price_forecast": 30 + rng.normal(0, 2, n),
            "spot_price_forecast": 50 + rng.normal(0, 10, n),
            "solar_forecast": np.clip(2000 * np.sin(np.pi * (hour - 7) / 13), 0, None),
            "wind_onshore_forecast": 2000 + rng.normal(0, 500, n),
            "wind_offshore_forecast": 800 + rng.normal(0, 300, n),
        },
        index=idx,
    )
    df.to_csv(path)


def test_train_and_log_model(tmp_path):
    tracking_uri = f"file://{tmp_path / 'mlruns'}"
    data_path = tmp_path / "sample.csv"
    _generate_test_data(data_path)

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

    result = train_and_log_model(config=config, data_path=data_path, tracking_uri=tracking_uri, horizon=1)

    assert result["run_id"]
    assert result["model_uri"].startswith("runs:/")
    assert result["metrics"]["mae"] > 0
