PYTHON := uv run python
PYTEST := uv run pytest
RUFF := uv run ruff
MLFLOW := uv run mlflow

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
REFERENCE_DATA ?= data/sample.csv
CURRENT_DATA ?= data/current.csv
DRIFT_OUTPUT ?= reports/drift_report.html
DRIFT_SHARE ?= 0.5

.PHONY: ensure-uv install install-aws install-monitoring lint test drift-check generate-data train train-all promote compare mlflow-ui serve package-model sagemaker-plan sagemaker-apply sagemaker-pipeline-plan sagemaker-pipeline-apply terraform-init terraform-validate

ensure-uv:
	@command -v uv >/dev/null 2>&1 || { echo "Installing uv..."; curl -LsSf https://astral.sh/uv/install.sh | sh; }

install: ensure-uv
	uv sync --frozen --extra dev

install-aws: ensure-uv
	uv sync --frozen --extra dev --extra aws --extra sagemaker

lint:
	$(RUFF) check src tests

format:
	$(RUFF) format src tests scripts

test:
	$(PYTEST) -q

generate-data:
	$(PYTHON) scripts/generate_sample_data.py

train:
	MLFLOW_TRACKING_URI="$(MLFLOW_TRACKING_URI)" $(PYTHON) -m mlops_serving_starter.training.train --config $(CONFIG) --data $(DATA) --tracking-uri "$(MLFLOW_TRACKING_URI)"

train-all:
	MLFLOW_TRACKING_URI="$(MLFLOW_TRACKING_URI)" $(PYTHON) -m mlops_serving_starter.training.train_all_horizons --config $(CONFIG) --data $(DATA) --tracking-uri "$(MLFLOW_TRACKING_URI)"

compare:
	MLFLOW_TRACKING_URI="$(MLFLOW_TRACKING_URI)" $(PYTHON) -m mlops_serving_starter.training.promote --compare --tracking-uri "$(MLFLOW_TRACKING_URI)"

promote:
	@if [ -z "$(VERSION)" ] || [ -z "$(ALIAS)" ]; then echo "Usage: make promote VERSION=4 ALIAS=production"; exit 1; fi
	MLFLOW_TRACKING_URI="$(MLFLOW_TRACKING_URI)" $(PYTHON) -m mlops_serving_starter.training.promote --version $(VERSION) --alias $(ALIAS) --tracking-uri "$(MLFLOW_TRACKING_URI)"

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

install-monitoring: ensure-uv
	uv sync --frozen --extra dev --extra monitoring

drift-check:
	$(PYTHON) -m mlops_serving_starter.monitoring.drift \
		--reference $(REFERENCE_DATA) \
		--current $(CURRENT_DATA) \
		--output $(DRIFT_OUTPUT) \
		--drift-share $(DRIFT_SHARE)

drift-check-ci:
	$(PYTHON) -m mlops_serving_starter.monitoring.drift \
		--reference $(REFERENCE_DATA) \
		--current $(CURRENT_DATA) \
		--output reports/drift_report.json \
		--json-only \
		--drift-share $(DRIFT_SHARE) \
		--fail-on-drift

