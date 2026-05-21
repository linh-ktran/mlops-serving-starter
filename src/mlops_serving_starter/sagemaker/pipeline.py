from __future__ import annotations

import argparse
import importlib
import json
from typing import Any

from mlops_serving_starter.utils.naming import sanitize_resource_name


def build_pipeline_definition(
    *,
    pipeline_name: str,
    role_arn: str,
    processing_image_uri: str,
    training_image_uri: str,
    inference_image_uri: str,
    processing_output_s3_uri: str,
    training_output_s3_uri: str,
    transform_output_s3_uri: str,
    input_data_s3_uri: str,
    transform_input_s3_uri: str,
    processing_instance_type: str = "ml.m5.large",
    training_instance_type: str = "ml.m5.large",
    transform_instance_type: str = "ml.m5.large",
) -> dict[str, Any]:
    safe_name = sanitize_resource_name(pipeline_name)

    return {
        "Version": "2020-12-01",
        "Metadata": {},
        "Parameters": [
            {"Name": "InputDataS3Uri", "Type": "String", "DefaultValue": input_data_s3_uri},
            {"Name": "TransformInputS3Uri", "Type": "String", "DefaultValue": transform_input_s3_uri},
        ],
        "Steps": [
            {
                "Name": "DataProcessing",
                "Type": "Processing",
                "Arguments": {
                    "ProcessingResources": {
                        "ClusterConfig": {
                            "InstanceType": processing_instance_type,
                            "InstanceCount": 1,
                            "VolumeSizeInGB": 30,
                        }
                    },
                    "AppSpecification": {"ImageUri": processing_image_uri},
                    "RoleArn": role_arn,
                    "ProcessingInputs": [
                        {
                            "InputName": "raw-data",
                            "AppManaged": False,
                            "S3Input": {
                                "S3Uri": {"Get": "Parameters.InputDataS3Uri"},
                                "LocalPath": "/opt/ml/processing/input",
                                "S3DataType": "S3Prefix",
                                "S3InputMode": "File",
                                "S3DataDistributionType": "FullyReplicated",
                                "S3CompressionType": "None",
                            },
                        }
                    ],
                    "ProcessingOutputConfig": {
                        "Outputs": [
                            {
                                "OutputName": "train",
                                "AppManaged": False,
                                "S3Output": {
                                    "S3Uri": processing_output_s3_uri,
                                    "LocalPath": "/opt/ml/processing/output/train",
                                    "S3UploadMode": "EndOfJob",
                                },
                            }
                        ]
                    },
                },
            },
            {
                "Name": "ModelTraining",
                "Type": "Training",
                "Arguments": {
                    "AlgorithmSpecification": {
                        "TrainingInputMode": "File",
                        "TrainingImage": training_image_uri,
                    },
                    "RoleArn": role_arn,
                    "OutputDataConfig": {"S3OutputPath": training_output_s3_uri},
                    "StoppingCondition": {"MaxRuntimeInSeconds": 3600},
                    "ResourceConfig": {
                        "VolumeSizeInGB": 30,
                        "InstanceCount": 1,
                        "InstanceType": training_instance_type,
                    },
                    "InputDataConfig": [
                        {
                            "DataSource": {
                                "S3DataSource": {
                                    "S3DataType": "S3Prefix",
                                    "S3Uri": {
                                        "Get": "Steps.DataProcessing.ProcessingOutputConfig.Outputs['train'].S3Output.S3Uri"
                                    },
                                    "S3DataDistributionType": "FullyReplicated",
                                }
                            },
                            "ContentType": "text/csv",
                            "ChannelName": "train",
                        }
                    ],
                },
            },
            {
                "Name": "CreateModel",
                "Type": "Model",
                "Arguments": {
                    "ExecutionRoleArn": role_arn,
                    "PrimaryContainer": {
                        "Image": inference_image_uri,
                        "Environment": {},
                        "ModelDataUrl": {"Get": "Steps.ModelTraining.ModelArtifacts.S3ModelArtifacts"},
                    },
                },
            },
            {
                "Name": "BatchTransform",
                "Type": "Transform",
                "Arguments": {
                    "ModelName": {"Get": "Steps.CreateModel.ModelName"},
                    "TransformInput": {
                        "ContentType": "text/csv",
                        "DataSource": {
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": {"Get": "Parameters.TransformInputS3Uri"},
                            }
                        },
                    },
                    "TransformOutput": {"S3OutputPath": transform_output_s3_uri},
                    "TransformResources": {"InstanceCount": 1, "InstanceType": transform_instance_type},
                },
            },
        ],
        "PipelineExperimentConfig": {
            "ExperimentName": safe_name,
            "TrialName": {"Get": "Execution.PipelineExecutionId"},
        },
    }


def build_upsert_pipeline_requests(
    *,
    pipeline_name: str,
    role_arn: str,
    pipeline_definition: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    definition_as_string = json.dumps(pipeline_definition, separators=(",", ":"), sort_keys=True)

    create_request = {
        "PipelineName": sanitize_resource_name(pipeline_name),
        "PipelineDisplayName": sanitize_resource_name(pipeline_name),
        "RoleArn": role_arn,
        "PipelineDefinition": definition_as_string,
    }
    update_request = {
        "PipelineName": sanitize_resource_name(pipeline_name),
        "RoleArn": role_arn,
        "PipelineDefinition": definition_as_string,
    }
    return {"create_pipeline": create_request, "update_pipeline": update_request}


def create_sagemaker_client(region_name: str | None = None):
    boto3 = importlib.import_module("boto3")
    return boto3.client("sagemaker", region_name=region_name)


def _error_code_from_exception(exc: Exception) -> str | None:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return None
    error = response.get("Error", {})
    if not isinstance(error, dict):
        return None
    code = error.get("Code")
    return code if isinstance(code, str) else None


def upsert_pipeline(
    *,
    sagemaker_client: Any,
    requests: dict[str, dict[str, Any]],
    apply: bool = False,
) -> dict[str, Any]:
    if not apply:
        return {"applied": False, "mode": "dry-run", "requests": requests}

    try:
        response = sagemaker_client.create_pipeline(**requests["create_pipeline"])
        action = "created"
    except Exception as exc:  # pragma: no cover
        code = _error_code_from_exception(exc)
        if code == "ResourceInUse":
            response = sagemaker_client.update_pipeline(**requests["update_pipeline"])
            action = "updated"
        else:
            raise

    return {
        "applied": True,
        "action": action,
        "requests": requests,
        "response": response,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or apply a SageMaker Pipeline definition")
    parser.add_argument("--pipeline-name", required=True)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--processing-image-uri", required=True)
    parser.add_argument("--training-image-uri", required=True)
    parser.add_argument("--inference-image-uri", required=True)
    parser.add_argument("--input-data-s3-uri", required=True)
    parser.add_argument("--transform-input-s3-uri", required=True)
    parser.add_argument("--processing-output-s3-uri", required=True)
    parser.add_argument("--training-output-s3-uri", required=True)
    parser.add_argument("--transform-output-s3-uri", required=True)
    parser.add_argument("--processing-instance-type", default="ml.m5.large")
    parser.add_argument("--training-instance-type", default="ml.m5.large")
    parser.add_argument("--transform-instance-type", default="ml.m5.large")
    parser.add_argument("--region", default=None)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    definition = build_pipeline_definition(
        pipeline_name=args.pipeline_name,
        role_arn=args.role_arn,
        processing_image_uri=args.processing_image_uri,
        training_image_uri=args.training_image_uri,
        inference_image_uri=args.inference_image_uri,
        processing_output_s3_uri=args.processing_output_s3_uri,
        training_output_s3_uri=args.training_output_s3_uri,
        transform_output_s3_uri=args.transform_output_s3_uri,
        input_data_s3_uri=args.input_data_s3_uri,
        transform_input_s3_uri=args.transform_input_s3_uri,
        processing_instance_type=args.processing_instance_type,
        training_instance_type=args.training_instance_type,
        transform_instance_type=args.transform_instance_type,
    )
    requests = build_upsert_pipeline_requests(
        pipeline_name=args.pipeline_name,
        role_arn=args.role_arn,
        pipeline_definition=definition,
    )

    if args.apply:
        client = create_sagemaker_client(region_name=args.region)
        result = upsert_pipeline(sagemaker_client=client, requests=requests, apply=True)
    else:
        result = upsert_pipeline(sagemaker_client=None, requests=requests, apply=False)

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
