# mlops-serving-starter
An end-to-end MLOps project covering training, experiment tracking, model serving, and deployment to AWS SageMaker.
Built to understand what happens after the notebook — how models get versioned, served, and deployed in production.
## Stack
- **MLflow** — experiment tracking and model registry
- **FastAPI** — REST API to serve predictions
- **SageMaker** — deployment target (endpoint + pipeline)
- **Terraform** — infrastructure for endpoint, schedule, alarms
- **GitHub Actions** — CI (lint, tests, Terraform validation)
## Project structure
```
src/mlops_serving_starter/
├── training/       # train + log to MLflow
├── serving/        # load MLflow model, run predictions
├── api/            # FastAPI /predict endpoint
└── sagemaker/      # inference handler, deploy script, pipeline builder
infra/terraform/    # SageMaker endpoint + EventBridge schedule + CloudWatch alarms
scripts/            # CLI for SageMaker deploy and pipeline steps
```
## Run it locally
```bash
make install
source .venv/bin/activate
# train a model and log it to MLflow
make train
# open MLflow UI to compare runs
make mlflow-ui   # → http://127.0.0.1:5001
# serve the model (use the run_id from training)
make serve MODEL_URI="runs:/<RUN_ID>/model"
# test the API
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"records":[{"sepal length (cm)":5.1,"sepal width (cm)":3.5,"petal length (cm)":1.4,"petal width (cm)":0.2}]}'
# run tests
make test
```
## Deploy to SageMaker
```bash
# package the MLflow model into model.tar.gz
make package-model MODEL_URI="runs:/<RUN_ID>/model"
# dry-run — prints the AWS request payloads, no API calls
make sagemaker-plan \
  IMAGE_URI=<ECR_IMAGE> \
  MODEL_DATA_URL=s3://<BUCKET>/artifacts/model.tar.gz \
  EXECUTION_ROLE_ARN=arn:aws:iam::<ACCOUNT>:role/<ROLE>
# apply — creates or updates the endpoint
make sagemaker-apply \
  IMAGE_URI=<ECR_IMAGE> \
  MODEL_DATA_URL=s3://<BUCKET>/artifacts/model.tar.gz \
  EXECUTION_ROLE_ARN=arn:aws:iam::<ACCOUNT>:role/<ROLE> \
  AWS_REGION=<REGION>
```
## SageMaker Pipeline
Generates a 4-step pipeline definition (processing → training → create model → batch transform) as JSON, ready to register in SageMaker.
```bash
make sagemaker-pipeline-plan \
  PIPELINE_NAME=my-pipeline \
  EXECUTION_ROLE_ARN=arn:aws:iam::<ACCOUNT>:role/<ROLE> \
  PROCESSING_IMAGE_URI=<ECR> TRAINING_IMAGE_URI=<ECR> INFERENCE_IMAGE_URI=<ECR> \
  INPUT_DATA_S3_URI=s3://<BUCKET>/input \
  TRANSFORM_INPUT_S3_URI=s3://<BUCKET>/transform-input \
  PROCESSING_OUTPUT_S3_URI=s3://<BUCKET>/processing-output \
  TRAINING_OUTPUT_S3_URI=s3://<BUCKET>/training-output \
  TRANSFORM_OUTPUT_S3_URI=s3://<BUCKET>/transform-output
```
## Infrastructure
```bash
make terraform-init
make terraform-validate
# to plan against a real account
cp infra/terraform/terraform.tfvars.example infra/terraform/terraform.tfvars
terraform -chdir=infra/terraform plan -var-file=terraform.tfvars
```
## What I'd add next
- Model promotion flow (Staging → Production) in MLflow before deploying
- Data drift monitoring with Evidently
- The EventBridge schedule wired to a real retrain trigger
