"""Data drift detection for aFRR capacity price forecast features.

Compares a reference dataset (training data) against current production data
to detect distribution shifts using Evidently.

Usage:
    python -m mlops_serving_starter.monitoring.drift \
        --reference data/reference.csv \
        --current data/current.csv \
        --output reports/drift_report.html

    # JSON-only (for CI / alerting):
    python -m mlops_serving_starter.monitoring.drift \
        --reference data/reference.csv \
        --current data/current.csv \
        --output reports/drift_report.json \
        --json-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from mlops_serving_starter.training.feature_engineering import (
    FORECAST_FEATURES,
    ROLLING_FEATURES,
)

logger = logging.getLogger(__name__)

# Features we want to monitor for drift
MONITORED_FEATURES: list[str] = (
    FORECAST_FEATURES
    + list(ROLLING_FEATURES.keys())
    + ["holiday_status", "weekend_status"]
)

# Default threshold: fraction of columns that must drift to flag dataset-level drift
DEFAULT_DRIFT_SHARE = 0.5


def load_dataset(path: Path) -> pd.DataFrame:
    """Load CSV and select monitored columns only."""
    df = pd.read_csv(path, parse_dates=["timestamp_utc"], index_col="timestamp_utc")
    df.sort_index(inplace=True)

    present = [col for col in MONITORED_FEATURES if col in df.columns]
    missing = [col for col in MONITORED_FEATURES if col not in df.columns]
    if missing:
        logger.warning("Missing columns (will be skipped): %s", missing)

    return df[present]


def build_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    drift_share: float = DEFAULT_DRIFT_SHARE,
):
    """Run Evidently data drift detection.

    Args:
        reference: Training/baseline data.
        current: Recent production data.
        drift_share: Fraction of columns that must drift to trigger dataset-level drift.

    Returns:
        Evidently Snapshot object with drift results.
    """
    report = Report([DataDriftPreset(drift_share=drift_share)])
    snapshot = report.run(reference_data=reference, current_data=current)
    return snapshot


def extract_drift_summary(snapshot) -> dict:
    """Extract a concise summary from the drift snapshot.

    Returns:
        Dict with overall drift status and per-feature results.
    """
    results = snapshot.metric_results

    drifted_count = 0
    drifted_share = 0.0
    total_columns = 0
    drifted_features: list[dict] = []
    drift_share_threshold = DEFAULT_DRIFT_SHARE

    for _key, metric_val in results.items():
        display = metric_val.display_name

        if "Count of Drifted Columns" in display:
            drifted_count = int(metric_val.count.value)
            drifted_share = float(metric_val.share.value)
            # Extract drift_share from params
            params = metric_val.metric_value_location.metric.params
            drift_share_threshold = params.get("drift_share", DEFAULT_DRIFT_SHARE)
        elif display.startswith("Value drift for "):
            total_columns += 1
            col_name = display.replace("Value drift for ", "")
            p_value = float(metric_val.value)

            # Extract threshold from metric params
            params = metric_val.metric_value_location.metric.params
            threshold = params.get("threshold", 0.05)

            if p_value < threshold:
                drifted_features.append(
                    {
                        "feature": col_name,
                        "stattest": params.get("method", ""),
                        "p_value": p_value,
                        "threshold": threshold,
                    }
                )

    # Dataset drift = share of drifted columns exceeds drift_share
    dataset_drift = drifted_share >= drift_share_threshold if total_columns > 0 else False

    return {
        "dataset_drift": dataset_drift,
        "share_of_drifted_columns": drifted_share,
        "number_of_drifted_columns": drifted_count,
        "total_columns": total_columns,
        "drifted_features": drifted_features,
    }


def run_drift_check(
    reference_path: Path,
    current_path: Path,
    output_path: Path | None = None,
    json_only: bool = False,
    drift_share: float = DEFAULT_DRIFT_SHARE,
) -> dict:
    """Full drift check pipeline: load data -> compute drift -> save report.

    Returns:
        Drift summary dict.
    """
    logger.info("Loading reference data from %s", reference_path)
    reference = load_dataset(reference_path)
    logger.info("Reference shape: %s", reference.shape)

    logger.info("Loading current data from %s", current_path)
    current = load_dataset(current_path)
    logger.info("Current shape: %s", current.shape)

    logger.info("Running drift detection (drift_share=%.2f)", drift_share)
    snapshot = build_drift_report(reference, current, drift_share=drift_share)

    summary = extract_drift_summary(snapshot)

    # Save outputs
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if json_only or output_path.suffix == ".json":
            json_path = output_path.with_suffix(".json")
            json_path.write_text(json.dumps(summary, indent=2, default=str))
            logger.info("Drift summary saved to %s", json_path)
        else:
            snapshot.save_html(str(output_path))
            logger.info("HTML report saved to %s", output_path)
            # Also save JSON summary alongside
            json_path = output_path.with_suffix(".json")
            json_path.write_text(json.dumps(summary, indent=2, default=str))
            logger.info("Drift summary saved to %s", json_path)

    # Log results
    if summary["dataset_drift"]:
        logger.warning(
            "\u26a0\ufe0f  DRIFT DETECTED — %d/%d features drifted: %s",
            summary["number_of_drifted_columns"],
            summary["total_columns"],
            [f["feature"] for f in summary["drifted_features"]],
        )
    else:
        logger.info("\u2705 No drift detected (%d features checked)", summary["total_columns"])

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data drift monitoring on forecast features")
    parser.add_argument("--reference", type=Path, required=True, help="Path to reference (training) CSV")
    parser.add_argument("--current", type=Path, required=True, help="Path to current (production) CSV")
    parser.add_argument(
        "--output", type=Path, default=Path("reports/drift_report.html"), help="Output report path"
    )
    parser.add_argument("--json-only", action="store_true", help="Save JSON summary only (no HTML)")
    parser.add_argument(
        "--drift-share", type=float, default=DEFAULT_DRIFT_SHARE,
        help="Fraction of drifted columns to trigger dataset drift (default: 0.5)"
    )
    parser.add_argument("--fail-on-drift", action="store_true", help="Exit with code 1 if drift detected")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    summary = run_drift_check(
        reference_path=args.reference,
        current_path=args.current,
        output_path=args.output,
        json_only=args.json_only,
        drift_share=args.drift_share,
    )

    if args.fail_on_drift and summary["dataset_drift"]:
        logger.error("Drift detected and --fail-on-drift is set. Exiting with code 1.")
        sys.exit(1)


if __name__ == "__main__":
    main()

