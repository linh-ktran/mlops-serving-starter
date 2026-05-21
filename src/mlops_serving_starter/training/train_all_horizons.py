"""Train models for all 4 horizons and log to MLflow.

Usage:
    .venv/bin/python -m mlops_serving_starter.training.train_all_horizons --data data/sample.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from mlops_serving_starter.training.train import load_training_config, train_and_log_model

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost models for all horizons (1-4)")
    parser.add_argument("--config", default="configs/train_config.json")
    parser.add_argument("--data", default="data/sample.csv")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--hour", type=int, default=None, help="Train for a specific hour (0-23)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = load_training_config(Path(args.config))

    results = []
    for horizon in range(1, 5):
        logger.info(f"{'=' * 60}")
        logger.info(f"Training horizon {horizon}/4")
        logger.info(f"{'=' * 60}")
        result = train_and_log_model(
            config=config,
            data_path=Path(args.data),
            tracking_uri=args.tracking_uri,
            horizon=horizon,
            hour=args.hour,
        )
        results.append({"horizon": horizon, **result})
        logger.info(f"Horizon {horizon}: MAE={result['metrics']['mae']:.2f}")

    print("\n" + "=" * 60)
    print("SUMMARY — All horizons")
    print("=" * 60)
    for r in results:
        print(
            f"  Horizon {r['horizon']}: MAE={r['metrics']['mae']:.2f}  RMSE={r['metrics']['rmse']:.2f}  run_id={r['run_id'][:8]}..."
        )
    print()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
