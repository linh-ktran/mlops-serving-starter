from mlops_serving_starter.sagemaker.deploy import (
    build_create_endpoint_config_request,
    build_create_model_request,
    deploy_to_sagemaker,
    sanitize_resource_name,
)


class _EndpointNotFound(Exception):
    def __init__(self):
        self.response = {"Error": {"Code": "ValidationException"}}
        super().__init__("Could not find endpoint")


class _FakeSageMakerClient:
    def __init__(self, *, endpoint_exists: bool = False):
        self.endpoint_exists = endpoint_exists
        self.calls = []

    def create_model(self, **kwargs):
        self.calls.append(("create_model", kwargs))
        return {"ModelArn": "arn:aws:sagemaker:region:acct:model/example"}

    def create_endpoint_config(self, **kwargs):
        self.calls.append(("create_endpoint_config", kwargs))
        return {"EndpointConfigArn": "arn:aws:sagemaker:region:acct:endpoint-config/example"}

    def create_endpoint(self, **kwargs):
        self.calls.append(("create_endpoint", kwargs))
        return {"EndpointArn": "arn:aws:sagemaker:region:acct:endpoint/example"}

    def describe_endpoint(self, **kwargs):
        self.calls.append(("describe_endpoint", kwargs))
        if not self.endpoint_exists:
            raise _EndpointNotFound()
        return {"EndpointStatus": "InService"}

    def update_endpoint(self, **kwargs):
        self.calls.append(("update_endpoint", kwargs))
        return {"EndpointArn": "arn:aws:sagemaker:region:acct:endpoint/example"}


def test_sanitize_resource_name_normalizes_special_characters():
    assert sanitize_resource_name("My Endpoint_Name !!!") == "My-Endpoint-Name"


def test_build_model_request_structure():
    request = build_create_model_request(
        endpoint_name="demo-endpoint",
        model_data_url="s3://bucket/model.tar.gz",
        image_uri="123456789012.dkr.ecr.eu-west-1.amazonaws.com/mlops-serving:latest",
        execution_role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
    )

    assert request["ModelName"] == "demo-endpoint-model"
    assert request["PrimaryContainer"]["Image"].endswith("mlops-serving:latest")
    assert request["PrimaryContainer"]["ModelDataUrl"] == "s3://bucket/model.tar.gz"


def test_build_endpoint_config_uses_instance_settings():
    request = build_create_endpoint_config_request(
        endpoint_name="demo-endpoint",
        instance_type="ml.c5.large",
        initial_instance_count=2,
        variant_name="Blue",
    )

    variant = request["ProductionVariants"][0]
    assert variant["InstanceType"] == "ml.c5.large"
    assert variant["InitialInstanceCount"] == 2
    assert variant["VariantName"] == "Blue"


def test_deploy_to_sagemaker_dry_run_returns_requests_only():
    client = _FakeSageMakerClient()

    result = deploy_to_sagemaker(
        sagemaker_client=client,
        endpoint_name="demo-endpoint",
        model_data_url="s3://bucket/model.tar.gz",
        image_uri="demo-image",
        execution_role_arn="demo-role",
        apply=False,
    )

    assert result["applied"] is False
    assert result["mode"] == "dry-run"
    assert client.calls == []


def test_deploy_to_sagemaker_apply_creates_endpoint_when_missing():
    client = _FakeSageMakerClient(endpoint_exists=False)

    result = deploy_to_sagemaker(
        sagemaker_client=client,
        endpoint_name="demo-endpoint",
        model_data_url="s3://bucket/model.tar.gz",
        image_uri="demo-image",
        execution_role_arn="demo-role",
        apply=True,
        update_existing=True,
    )

    assert result["applied"] is True
    assert result["endpoint_action"] == "created"
    assert [name for name, _ in client.calls] == [
        "create_model",
        "create_endpoint_config",
        "describe_endpoint",
        "create_endpoint",
    ]


def test_deploy_to_sagemaker_apply_updates_existing_endpoint():
    client = _FakeSageMakerClient(endpoint_exists=True)

    result = deploy_to_sagemaker(
        sagemaker_client=client,
        endpoint_name="demo-endpoint",
        model_data_url="s3://bucket/model.tar.gz",
        image_uri="demo-image",
        execution_role_arn="demo-role",
        apply=True,
        update_existing=True,
    )

    assert result["endpoint_action"] == "updated"
    assert [name for name, _ in client.calls] == [
        "create_model",
        "create_endpoint_config",
        "describe_endpoint",
        "update_endpoint",
    ]
