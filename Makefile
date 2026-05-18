VENV ?= .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff
MLFLOW := $(VENV)/bin/mlflow

CONFIG ?= configs/train_config.json
DATA ?= data/sample.csv
MLFLOW_TRACKING_URI ?= file:$(CURDIR)/mlruns
MODEL_URI ?=
MODEL_OUTPUT ?= artifacts/model.tar.gz
ENDPOINT_NAME ?= mlops-serving-starter-endpoint
IMAGE_URI ?=
MODEL_DATA_URL ?=
EXECUTION_ROLE_ARN ?=
INSTANCE_TYPE ?= ml.m5.large
INITIAL_INSTANCE_COUNT ?= 1
AWS_REGION ?=
PIPELINE_NAME ?= mlops-serving-starter-pipeline
PROCESSING_IMAGE_URI ?=
TRAINING_IMAGE_URI ?=
INFERENCE_IMAGE_URI ?=
INPUT_DATA_S3_URI ?=
TRANSFORM_INPUT_S3_URI ?=
PROCESSING_OUTPUT_S3_URI ?=
TRAINING_OUTPUT_S3_URI ?=
TRANSFORM_OUTPUT_S3_URI ?=

.PHONY: install install-aws lint test generate-data train mlflow-ui serve package-model sagemaker-plan sagemaker-apply sagemaker-pipeline-plan sagemaker-pipeline-apply terraform-init terraform-validate

install:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

install-aws:
	$(PIP) install -e ".[dev,aws,sagemaker]"

lint:
	$(RUFF) check src tests

test:
	$(PYTEST) -q

generate-data:
	$(PYTHON) scripts/generate_sample_data.py

train:
	MLFLOW_TRACKING_URI="$(MLFLOW_TRACKING_URI)" $(PYTHON) -m mlops_serving_starter.training.train --config $(CONFIG) --data $(DATA) --tracking-uri "$(MLFLOW_TRACKING_URI)"

mlflow-ui:
	MLFLOW_TRACKING_URI="$(MLFLOW_TRACKING_URI)" $(MLFLOW) ui --host 127.0.0.1 --port 5001

serve:
	@if [ -z "$(MODEL_URI)" ]; then echo "MODEL_URI is required. Example: make serve MODEL_URI=runs:/<RUN_ID>/model"; exit 1; fi
	MLFLOW_TRACKING_URI="$(MLFLOW_TRACKING_URI)" MODEL_URI="$(MODEL_URI)" $(PYTHON) -m mlops_serving_starter.api.app

package-model:
	$(PYTHON) scripts/package_model_for_sagemaker.py --model-uri $(MODEL_URI) --output $(MODEL_OUTPUT)

sagemaker-plan:
	@if [ -z "$(IMAGE_URI)" ] || [ -z "$(MODEL_DATA_URL)" ] || [ -z "$(EXECUTION_ROLE_ARN)" ]; then echo "IMAGE_URI, MODEL_DATA_URL, and EXECUTION_ROLE_ARN are required."; exit 1; fi
	$(PYTHON) scripts/sagemaker_deploy_example.py --endpoint-name $(ENDPOINT_NAME) --image-uri $(IMAGE_URI) --model-data-url $(MODEL_DATA_URL) --execution-role-arn $(EXECUTION_ROLE_ARN) --instance-type $(INSTANCE_TYPE) --initial-instance-count $(INITIAL_INSTANCE_COUNT) $(if $(AWS_REGION),--region $(AWS_REGION),)

sagemaker-apply:
	@if [ -z "$(IMAGE_URI)" ] || [ -z "$(MODEL_DATA_URL)" ] || [ -z "$(EXECUTION_ROLE_ARN)" ]; then echo "IMAGE_URI, MODEL_DATA_URL, and EXECUTION_ROLE_ARN are required."; exit 1; fi
	$(PYTHON) scripts/sagemaker_deploy_example.py --apply --update-existing --endpoint-name $(ENDPOINT_NAME) --image-uri $(IMAGE_URI) --model-data-url $(MODEL_DATA_URL) --execution-role-arn $(EXECUTION_ROLE_ARN) --instance-type $(INSTANCE_TYPE) --initial-instance-count $(INITIAL_INSTANCE_COUNT) $(if $(AWS_REGION),--region $(AWS_REGION),)

sagemaker-pipeline-plan:
	@if [ -z "$(EXECUTION_ROLE_ARN)" ] || [ -z "$(PROCESSING_IMAGE_URI)" ] || [ -z "$(TRAINING_IMAGE_URI)" ] || [ -z "$(INFERENCE_IMAGE_URI)" ] || [ -z "$(INPUT_DATA_S3_URI)" ] || [ -z "$(TRANSFORM_INPUT_S3_URI)" ] || [ -z "$(PROCESSING_OUTPUT_S3_URI)" ] || [ -z "$(TRAINING_OUTPUT_S3_URI)" ] || [ -z "$(TRANSFORM_OUTPUT_S3_URI)" ]; then echo "Missing required vars: EXECUTION_ROLE_ARN, PROCESSING_IMAGE_URI, TRAINING_IMAGE_URI, INFERENCE_IMAGE_URI, INPUT_DATA_S3_URI, TRANSFORM_INPUT_S3_URI, PROCESSING_OUTPUT_S3_URI, TRAINING_OUTPUT_S3_URI, TRANSFORM_OUTPUT_S3_URI"; exit 1; fi
	$(PYTHON) scripts/sagemaker_pipeline_example.py --pipeline-name $(PIPELINE_NAME) --role-arn $(EXECUTION_ROLE_ARN) --processing-image-uri $(PROCESSING_IMAGE_URI) --training-image-uri $(TRAINING_IMAGE_URI) --inference-image-uri $(INFERENCE_IMAGE_URI) --input-data-s3-uri $(INPUT_DATA_S3_URI) --transform-input-s3-uri $(TRANSFORM_INPUT_S3_URI) --processing-output-s3-uri $(PROCESSING_OUTPUT_S3_URI) --training-output-s3-uri $(TRAINING_OUTPUT_S3_URI) --transform-output-s3-uri $(TRANSFORM_OUTPUT_S3_URI) $(if $(AWS_REGION),--region $(AWS_REGION),)

sagemaker-pipeline-apply:
	@if [ -z "$(EXECUTION_ROLE_ARN)" ] || [ -z "$(PROCESSING_IMAGE_URI)" ] || [ -z "$(TRAINING_IMAGE_URI)" ] || [ -z "$(INFERENCE_IMAGE_URI)" ] || [ -z "$(INPUT_DATA_S3_URI)" ] || [ -z "$(TRANSFORM_INPUT_S3_URI)" ] || [ -z "$(PROCESSING_OUTPUT_S3_URI)" ] || [ -z "$(TRAINING_OUTPUT_S3_URI)" ] || [ -z "$(TRANSFORM_OUTPUT_S3_URI)" ]; then echo "Missing required vars: EXECUTION_ROLE_ARN, PROCESSING_IMAGE_URI, TRAINING_IMAGE_URI, INFERENCE_IMAGE_URI, INPUT_DATA_S3_URI, TRANSFORM_INPUT_S3_URI, PROCESSING_OUTPUT_S3_URI, TRAINING_OUTPUT_S3_URI, TRANSFORM_OUTPUT_S3_URI"; exit 1; fi
	$(PYTHON) scripts/sagemaker_pipeline_example.py --apply --pipeline-name $(PIPELINE_NAME) --role-arn $(EXECUTION_ROLE_ARN) --processing-image-uri $(PROCESSING_IMAGE_URI) --training-image-uri $(TRAINING_IMAGE_URI) --inference-image-uri $(INFERENCE_IMAGE_URI) --input-data-s3-uri $(INPUT_DATA_S3_URI) --transform-input-s3-uri $(TRANSFORM_INPUT_S3_URI) --processing-output-s3-uri $(PROCESSING_OUTPUT_S3_URI) --training-output-s3-uri $(TRAINING_OUTPUT_S3_URI) --transform-output-s3-uri $(TRANSFORM_OUTPUT_S3_URI) $(if $(AWS_REGION),--region $(AWS_REGION),)

terraform-init:
	terraform -chdir=infra/terraform init -backend=false

terraform-validate:
	terraform -chdir=infra/terraform validate

