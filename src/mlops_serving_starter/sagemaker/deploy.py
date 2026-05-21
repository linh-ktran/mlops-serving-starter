from __future__ import annotations

import argparse
import importlib
import json
from typing import Any

from mlops_serving_starter.utils.naming import sanitize_resource_name

DEFAULT_INSTANCE_TYPE = "ml.m5.large"
DEFAULT_INITIAL_INSTANCE_COUNT = 1
DEFAULT_VARIANT_NAME = "AllTraffic"


def build_resource_names(endpoint_name: str) -> dict[str, str]:
    endpoint_name = sanitize_resource_name(endpoint_name)
    return {
        "endpoint_name": endpoint_name,
        "model_name": sanitize_resource_name(f"{endpoint_name}-model"),
        "endpoint_config_name": sanitize_resource_name(f"{endpoint_name}-config"),
    }


def build_create_model_request(
    *,
    endpoint_name: str,
    model_data_url: str,
    image_uri: str,
    execution_role_arn: str,
) -> dict[str, Any]:
    names = build_resource_names(endpoint_name)
    return {
        "ModelName": names["model_name"],
        "PrimaryContainer": {
            "Image": image_uri,
            "ModelDataUrl": model_data_url,
            "Environment": {},
        },
        "ExecutionRoleArn": execution_role_arn,
    }


def build_create_endpoint_config_request(
    *,
    endpoint_name: str,
    instance_type: str = DEFAULT_INSTANCE_TYPE,
    initial_instance_count: int = DEFAULT_INITIAL_INSTANCE_COUNT,
    variant_name: str = DEFAULT_VARIANT_NAME,
) -> dict[str, Any]:
    names = build_resource_names(endpoint_name)
    return {
        "EndpointConfigName": names["endpoint_config_name"],
        "ProductionVariants": [
            {
                "VariantName": variant_name,
                "ModelName": names["model_name"],
                "InitialInstanceCount": initial_instance_count,
                "InstanceType": instance_type,
                "InitialVariantWeight": 1.0,
            }
        ],
    }


def build_create_endpoint_request(*, endpoint_name: str) -> dict[str, Any]:
    names = build_resource_names(endpoint_name)
    return {
        "EndpointName": names["endpoint_name"],
        "EndpointConfigName": names["endpoint_config_name"],
    }


def _looks_like_not_found_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    error_code = None
    if isinstance(response, dict):
        error_code = response.get("Error", {}).get("Code")
    text = str(exc)
    return error_code in {"ValidationException", "ResourceNotFound", "ResourceNotFoundException"} or (
        "Could not find endpoint" in text
    )


def create_sagemaker_client(region_name: str | None = None):
    boto3 = importlib.import_module("boto3")
    return boto3.client("sagemaker", region_name=region_name)


def deploy_to_sagemaker(
    *,
    sagemaker_client,
    endpoint_name: str,
    model_data_url: str,
    image_uri: str,
    execution_role_arn: str,
    instance_type: str = DEFAULT_INSTANCE_TYPE,
    initial_instance_count: int = DEFAULT_INITIAL_INSTANCE_COUNT,
    variant_name: str = DEFAULT_VARIANT_NAME,
    apply: bool = False,
    update_existing: bool = False,
) -> dict[str, Any]:
    model_request = build_create_model_request(
        endpoint_name=endpoint_name,
        model_data_url=model_data_url,
        image_uri=image_uri,
        execution_role_arn=execution_role_arn,
    )
    endpoint_config_request = build_create_endpoint_config_request(
        endpoint_name=endpoint_name,
        instance_type=instance_type,
        initial_instance_count=initial_instance_count,
        variant_name=variant_name,
    )
    endpoint_request = build_create_endpoint_request(endpoint_name=endpoint_name)

    requests = {
        "create_model": model_request,
        "create_endpoint_config": endpoint_config_request,
        "create_endpoint": endpoint_request,
    }

    if not apply:
        return {
            "applied": False,
            "mode": "dry-run",
            "requests": requests,
        }

    model_response = sagemaker_client.create_model(**model_request)
    endpoint_config_response = sagemaker_client.create_endpoint_config(**endpoint_config_request)

    endpoint_action = "created"
    if update_existing:
        try:
            sagemaker_client.describe_endpoint(EndpointName=endpoint_request["EndpointName"])
        except Exception as exc:  # pragma: no cover - exercised by unit test fakes
            if _looks_like_not_found_error(exc):
                endpoint_response = sagemaker_client.create_endpoint(**endpoint_request)
            else:
                raise
        else:
            endpoint_response = sagemaker_client.update_endpoint(**endpoint_request)
            endpoint_action = "updated"
    else:
        endpoint_response = sagemaker_client.create_endpoint(**endpoint_request)

    return {
        "applied": True,
        "endpoint_action": endpoint_action,
        "requests": requests,
        "responses": {
            "create_model": model_response,
            "create_endpoint_config": endpoint_config_response,
            "create_endpoint": endpoint_response,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or update a SageMaker endpoint for this project")
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--model-data-url", required=True, help="S3 URI to model.tar.gz")
    parser.add_argument("--image-uri", required=True, help="ECR image URI for inference container")
    parser.add_argument("--execution-role-arn", required=True)
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    parser.add_argument("--initial-instance-count", type=int, default=DEFAULT_INITIAL_INSTANCE_COUNT)
    parser.add_argument("--variant-name", default=DEFAULT_VARIANT_NAME)
    parser.add_argument("--region", default=None)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually call AWS APIs. Without this flag the script prints the requests only.",
    )
    parser.add_argument(
        "--update-existing",
        action="store_true",
        help="If the endpoint exists, update it instead of failing on create.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.apply:
        client = create_sagemaker_client(region_name=args.region)
        result = deploy_to_sagemaker(
            sagemaker_client=client,
            endpoint_name=args.endpoint_name,
            model_data_url=args.model_data_url,
            image_uri=args.image_uri,
            execution_role_arn=args.execution_role_arn,
            instance_type=args.instance_type,
            initial_instance_count=args.initial_instance_count,
            variant_name=args.variant_name,
            apply=True,
            update_existing=args.update_existing,
        )
    else:
        result = deploy_to_sagemaker(
            sagemaker_client=object(),
            endpoint_name=args.endpoint_name,
            model_data_url=args.model_data_url,
            image_uri=args.image_uri,
            execution_role_arn=args.execution_role_arn,
            instance_type=args.instance_type,
            initial_instance_count=args.initial_instance_count,
            variant_name=args.variant_name,
            apply=False,
            update_existing=args.update_existing,
        )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
