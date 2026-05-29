"""Tests for data drift monitoring module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("evidently", reason="evidently not installed (install with: uv sync --extra monitoring)")

from mlops_serving_starter.monitoring.drift import (  # noqa: E402
    MONITORED_FEATURES,
    build_drift_report,
    extract_drift_summary,
    load_dataset,
    run_drift_check,
)


@pytest.fixture
def sample_reference_data(tmp_path: Path) -> Path:
    """Create a reference CSV with known distributions."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2025-01-01", periods=n, freq="h")
    data = {
        "timestamp_utc": dates,
        "fcr_price_symmetric": np.random.normal(15.0, 3.0, n),
        "consumption_forecast": np.random.normal(52000, 5000, n),
        "gas_price_forecast": np.random.normal(32.0, 5.0, n),
        "spot_price_forecast": np.random.normal(55.0, 10.0, n),
        "solar_forecast": np.random.normal(2500, 800, n),
        "wind_onshore_forecast": np.random.normal(2100, 600, n),
        "wind_offshore_forecast": np.random.normal(900, 300, n),
        "rolling_mean_7": np.random.normal(23.0, 4.0, n),
        "rolling_min_7": np.random.normal(18.0, 3.0, n),
        "rolling_max_7": np.random.normal(30.0, 5.0, n),
        "holiday_status": np.random.choice([0, 1], n, p=[0.9, 0.1]),
        "weekend_status": np.random.choice([0, 1], n, p=[0.71, 0.29]),
    }
    df = pd.DataFrame(data)
    path = tmp_path / "reference.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_current_no_drift(tmp_path: Path) -> Path:
    """Create current data with same distribution (no drift)."""
    np.random.seed(99)
    n = 200
    dates = pd.date_range("2025-01-22", periods=n, freq="h")
    data = {
        "timestamp_utc": dates,
        "fcr_price_symmetric": np.random.normal(15.0, 3.0, n),
        "consumption_forecast": np.random.normal(52000, 5000, n),
        "gas_price_forecast": np.random.normal(32.0, 5.0, n),
        "spot_price_forecast": np.random.normal(55.0, 10.0, n),
        "solar_forecast": np.random.normal(2500, 800, n),
        "wind_onshore_forecast": np.random.normal(2100, 600, n),
        "wind_offshore_forecast": np.random.normal(900, 300, n),
        "rolling_mean_7": np.random.normal(23.0, 4.0, n),
        "rolling_min_7": np.random.normal(18.0, 3.0, n),
        "rolling_max_7": np.random.normal(30.0, 5.0, n),
        "holiday_status": np.random.choice([0, 1], n, p=[0.9, 0.1]),
        "weekend_status": np.random.choice([0, 1], n, p=[0.71, 0.29]),
    }
    df = pd.DataFrame(data)
    path = tmp_path / "current_no_drift.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_current_with_drift(tmp_path: Path) -> Path:
    """Create current data with shifted distributions (drift)."""
    np.random.seed(123)
    n = 200
    dates = pd.date_range("2025-01-22", periods=n, freq="h")
    data = {
        "timestamp_utc": dates,
        # Shifted means to trigger drift
        "fcr_price_symmetric": np.random.normal(25.0, 3.0, n),  # was 15
        "consumption_forecast": np.random.normal(62000, 5000, n),  # was 52000
        "gas_price_forecast": np.random.normal(50.0, 5.0, n),  # was 32
        "spot_price_forecast": np.random.normal(80.0, 10.0, n),  # was 55
        "solar_forecast": np.random.normal(500, 200, n),  # was 2500
        "wind_onshore_forecast": np.random.normal(4500, 600, n),  # was 2100
        "wind_offshore_forecast": np.random.normal(2500, 300, n),  # was 900
        "rolling_mean_7": np.random.normal(35.0, 4.0, n),  # was 23
        "rolling_min_7": np.random.normal(30.0, 3.0, n),  # was 18
        "rolling_max_7": np.random.normal(50.0, 5.0, n),  # was 30
        "holiday_status": np.random.choice([0, 1], n, p=[0.9, 0.1]),
        "weekend_status": np.random.choice([0, 1], n, p=[0.71, 0.29]),
    }
    df = pd.DataFrame(data)
    path = tmp_path / "current_drifted.csv"
    df.to_csv(path, index=False)
    return path


class TestLoadDataset:
    def test_loads_monitored_columns(self, sample_reference_data: Path):
        df = load_dataset(sample_reference_data)
        assert len(df) == 500
        # All monitored features should be present
        for col in MONITORED_FEATURES:
            assert col in df.columns

    def test_handles_missing_columns(self, tmp_path: Path):
        """Should skip missing columns gracefully."""
        dates = pd.date_range("2025-01-01", periods=10, freq="h")
        df = pd.DataFrame({"timestamp_utc": dates, "fcr_price_symmetric": range(10)})
        path = tmp_path / "partial.csv"
        df.to_csv(path, index=False)

        result = load_dataset(path)
        assert "fcr_price_symmetric" in result.columns
        assert len(result.columns) == 1


class TestBuildDriftReport:
    def test_no_drift_same_distribution(self, sample_reference_data, sample_current_no_drift):
        ref = load_dataset(sample_reference_data)
        cur = load_dataset(sample_current_no_drift)
        snapshot = build_drift_report(ref, cur)
        summary = extract_drift_summary(snapshot)
        # Same distribution should not trigger dataset drift
        assert summary["dataset_drift"] is False

    def test_drift_detected_shifted_distribution(self, sample_reference_data, sample_current_with_drift):
        ref = load_dataset(sample_reference_data)
        cur = load_dataset(sample_current_with_drift)
        snapshot = build_drift_report(ref, cur)
        summary = extract_drift_summary(snapshot)
        # Should detect drift in multiple features
        assert summary["dataset_drift"] is True
        assert summary["number_of_drifted_columns"] > 0
        drifted_names = [f["feature"] for f in summary["drifted_features"]]
        # These features have huge mean shifts
        assert "fcr_price_symmetric" in drifted_names
        assert "gas_price_forecast" in drifted_names


class TestRunDriftCheck:
    def test_saves_html_report(self, sample_reference_data, sample_current_no_drift, tmp_path):
        output = tmp_path / "report.html"
        summary = run_drift_check(
            reference_path=sample_reference_data,
            current_path=sample_current_no_drift,
            output_path=output,
        )
        assert output.exists()
        assert (tmp_path / "report.json").exists()
        assert "dataset_drift" in summary

    def test_saves_json_only(self, sample_reference_data, sample_current_with_drift, tmp_path):
        output = tmp_path / "report.json"
        summary = run_drift_check(
            reference_path=sample_reference_data,
            current_path=sample_current_with_drift,
            output_path=output,
            json_only=True,
        )
        assert output.exists()
        assert summary["dataset_drift"] is True
