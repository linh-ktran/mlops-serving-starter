import json

from mlops_serving_starter.sagemaker.pipeline import (
    build_pipeline_definition,
    build_upsert_pipeline_requests,
    sanitize_resource_name,
    upsert_pipeline,
)


class _ResourceInUseError(Exception):
    def __init__(self):
        self.response = {"Error": {"Code": "ResourceInUse"}}
        super().__init__("Pipeline already exists")


class _FakeSageMakerClient:
    def __init__(self, *, exists: bool = False):
        self.exists = exists
        self.calls = []

    def create_pipeline(self, **kwargs):
        self.calls.append(("create_pipeline", kwargs))
        if self.exists:
            raise _ResourceInUseError()
        return {"PipelineArn": "arn:aws:sagemaker:region:acct:pipeline/example"}

    def update_pipeline(self, **kwargs):
        self.calls.append(("update_pipeline", kwargs))
        return {"PipelineArn": "arn:aws:sagemaker:region:acct:pipeline/example"}


def _sample_definition():
    return build_pipeline_definition(
        pipeline_name="demo-pipeline",
        role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
        processing_image_uri="111111111111.dkr.ecr.eu-west-1.amazonaws.com/preprocess:latest",
        training_image_uri="111111111111.dkr.ecr.eu-west-1.amazonaws.com/train:latest",
        inference_image_uri="111111111111.dkr.ecr.eu-west-1.amazonaws.com/infer:latest",
        processing_output_s3_uri="s3://bucket/processing",
        training_output_s3_uri="s3://bucket/training",
        transform_output_s3_uri="s3://bucket/inference",
        input_data_s3_uri="s3://bucket/input",
        transform_input_s3_uri="s3://bucket/transform-input",
    )


def test_sanitize_resource_name_for_pipeline():
    assert sanitize_resource_name("demo pipeline___name") == "demo-pipeline-name"


def test_pipeline_definition_contains_expected_step_order():
    definition = _sample_definition()

    assert [step["Name"] for step in definition["Steps"]] == [
        "DataProcessing",
        "ModelTraining",
        "CreateModel",
        "BatchTransform",
    ]


def test_build_upsert_requests_include_serialized_definition():
    definition = _sample_definition()
    requests = build_upsert_pipeline_requests(
        pipeline_name="demo-pipeline",
        role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
        pipeline_definition=definition,
    )

    payload = json.loads(requests["create_pipeline"]["PipelineDefinition"])
    assert payload["Version"] == "2020-12-01"
    assert requests["create_pipeline"]["PipelineName"] == "demo-pipeline"


def test_upsert_pipeline_dry_run_returns_requests_only():
    requests = {
        "create_pipeline": {"PipelineName": "demo"},
        "update_pipeline": {"PipelineName": "demo"},
    }

    result = upsert_pipeline(sagemaker_client=object(), requests=requests, apply=False)

    assert result["applied"] is False
    assert result["mode"] == "dry-run"


def test_upsert_pipeline_apply_creates_when_missing():
    requests = {
        "create_pipeline": {"PipelineName": "demo"},
        "update_pipeline": {"PipelineName": "demo"},
    }
    client = _FakeSageMakerClient(exists=False)

    result = upsert_pipeline(sagemaker_client=client, requests=requests, apply=True)

    assert result["action"] == "created"
    assert [name for name, _ in client.calls] == ["create_pipeline"]


def test_upsert_pipeline_apply_updates_when_exists():
    requests = {
        "create_pipeline": {"PipelineName": "demo"},
        "update_pipeline": {"PipelineName": "demo"},
    }
    client = _FakeSageMakerClient(exists=True)

    result = upsert_pipeline(sagemaker_client=client, requests=requests, apply=True)

    assert result["action"] == "updated"
    assert [name for name, _ in client.calls] == ["create_pipeline", "update_pipeline"]
