"""Manage model lifecycle: promote versions using MLflow aliases.

Usage:
    python -m mlops_serving_starter.training.promote --compare
    python -m mlops_serving_starter.training.promote --version 4 --alias production
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "afrr-capacity-price-xgboost"


def get_client(tracking_uri: str | None = None) -> MlflowClient:
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", f"file://{os.getcwd()}/mlruns")
    mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri)


def list_versions(client: MlflowClient, model_name: str) -> list[dict]:
    """List all versions of a registered model with their aliases and metrics."""
    versions = client.search_model_versions(f"name='{model_name}'")
    results = []
    for v in sorted(versions, key=lambda x: int(x.version)):
        run = client.get_run(v.run_id)
        results.append(
            {
                "version": int(v.version),
                "run_id": v.run_id,
                "aliases": v.aliases if hasattr(v, "aliases") else [],
                "metrics": {k: round(val, 4) for k, val in run.data.metrics.items()},
                "params": {
                    "target": run.data.params.get("target", "?"),
                    "horizon": run.data.params.get("horizon", "?"),
                    "hour": run.data.params.get("hour", "?"),
                },
                "created": str(v.creation_timestamp),
            }
        )
    return results


def promote_version(client: MlflowClient, model_name: str, version: int, alias: str) -> None:
    """Set an alias (e.g. 'staging', 'production') on a model version."""
    client.set_registered_model_alias(model_name, alias, version=version)
    logger.info(f"Model '{model_name}' version {version} → @{alias}")


def compare_versions(client: MlflowClient, model_name: str) -> None:
    """Print a comparison table of all model versions."""
    versions = list_versions(client, model_name)
    if not versions:
        print(f"No versions found for model '{model_name}'")
        return

    # Header
    print(f"\n{'Ver':<5} {'Target':<28} {'H':<3} {'Hour':<5} {'MAE':<8} {'RMSE':<8} {'Aliases'}")
    print("-" * 80)
    for v in versions:
        aliases_str = ", ".join(v["aliases"]) if v["aliases"] else ""
        mae = v["metrics"].get("mae", "?")
        rmse = v["metrics"].get("rmse", "?")
        print(
            f"{v['version']:<5} "
            f"{v['params']['target']:<28} "
            f"{v['params']['horizon']:<3} "
            f"{v['params']['hour']:<5} "
            f"{mae:<8} "
            f"{rmse:<8} "
            f"{aliases_str}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote MLflow model versions with aliases")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--list", action="store_true", help="List all model versions (JSON)")
    parser.add_argument("--compare", action="store_true", help="Compare all versions in a table")
    parser.add_argument("--version", type=int, help="Model version to promote")
    parser.add_argument(
        "--alias",
        choices=["staging", "production", "champion", "challenger"],
        help="Alias to set on the version",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    client = get_client(args.tracking_uri)

    if args.list:
        versions = list_versions(client, args.model_name)
        print(json.dumps(versions, indent=2, default=str))
    elif args.compare:
        compare_versions(client, args.model_name)
    elif args.version and args.alias:
        promote_version(client, args.model_name, args.version, args.alias)
        print(f"✓ Model '{args.model_name}' v{args.version} is now @{args.alias}")
        print(f'  Serve it with: make serve MODEL_URI="models:/{args.model_name}@{args.alias}"')
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
