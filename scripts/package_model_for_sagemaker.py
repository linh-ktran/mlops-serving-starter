from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import mlflow.artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an MLflow model to model.tar.gz for SageMaker")
    parser.add_argument("--model-uri", required=True, help="Example: runs:/<RUN_ID>/model")
    parser.add_argument("--output", default="artifacts/model.tar.gz")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    local_model_dir = Path(mlflow.artifacts.download_artifacts(artifact_uri=args.model_uri))

    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(local_model_dir, arcname="model")

    print(f"Created {output_path}")


if __name__ == "__main__":
    main()
