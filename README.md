# mlops-serving-starter
An end-to-end MLOps project covering training, experiment tracking, model serving, and deployment to AWS SageMaker.
Built to understand what happens after the notebook — how models get versioned, served, and deployed in production.

## Use Case: aFRR Capacity Price Forecasting
This project forecasts **French aFRR capacity prices** using XGBoost time-series regression — inspired by a real production pipeline.

**Problem:** Predict hourly capacity prices (EUR/MW) for up to 4 days ahead (horizons 1–4).

**Approach:**
- XGBoost regression with early stopping and bias correction
- Lag features (1, 2, 3, 7, 14, 28 days), rolling statistics (7-day mean/min/max)
- Exogenous inputs: FCR prices, consumption/gas/spot/solar/wind forecasts
- Calendar features: French holidays, weekends
- Per-horizon models (each horizon has its own trained model)
- Chronological train/val split (no data leakage)

## Stack
- **uv** — fast Python package manager and project tool
- **Hatchling** — PEP 517 build backend
- **Ruff** — linter and formatter
- **XGBoost** — time-series regression models
- **MLflow** — experiment tracking and model registry
- **FastAPI** — REST API to serve predictions
- **SageMaker** — deployment target (endpoint + pipeline)
- **Terraform** — infrastructure for endpoint, schedule, alarms
- **GitHub Actions** — CI (lint, tests, Terraform validation)

## Project structure
```
src/mlops_serving_starter/
├── training/
│   ├── train.py                # train XGBoost + log to MLflow
│   └── feature_engineering.py  # lags, rolling stats, calendar, exogenous features
├── serving/        # load MLflow model, run predictions
├── api/            # FastAPI /predict endpoint
├── monitoring/     # data drift detection with Evidently
└── sagemaker/      # inference handler, deploy script, pipeline builder
scripts/
├── generate_sample_data.py     # generate synthetic aFRR-like data
configs/
├── train_config.json           # model hyperparameters & experiment config
infra/terraform/    # SageMaker endpoint + EventBridge schedule + CloudWatch alarms
```

## Run it locally

Requires [uv](https://docs.astral.sh/uv/). See [installation options](https://docs.astral.sh/uv/getting-started/installation/) (Homebrew, pipx, standalone installer, etc.).

```bash
make install          # uv sync --extra dev (creates .venv automatically)

# generate synthetic time-series data
make generate-data

# train a model for horizon 1 (day-ahead) and log it to MLflow
make train

# train for a specific hour (like the production system)
uv run python -m mlops_serving_starter.training.train --data data/sample.csv --horizon 1 --hour 10

# open MLflow UI to compare runs
make mlflow-ui   # → http://127.0.0.1:5001

# serve the model (use the run_id from training)
make serve MODEL_URI="runs:/<RUN_ID>/model"

# test the API (send pre-computed feature vector)
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"records":[{"afrr_capacity_price_up_lag_1_h1":25.3,"afrr_capacity_price_up_lag_2_h1":24.1,"afrr_capacity_price_up_lag_3_h1":23.5,"afrr_capacity_price_up_lag_7_h1":22.0,"afrr_capacity_price_up_lag_14_h1":21.5,"afrr_capacity_price_up_lag_28_h1":20.0,"rolling_mean_7":23.4,"rolling_min_7":18.2,"rolling_max_7":30.1,"fcr_price_symmetric":15.0,"consumption_forecast":52000,"gas_price_forecast":32.5,"spot_price_forecast":55.0,"solar_forecast":2500,"wind_onshore_forecast":2100,"wind_offshore_forecast":900,"holiday_status":0,"weekend_status":0}]}'

# run tests
make test
```

## Model Lifecycle (MLflow)
```bash
# train all 4 horizons at once
make train-all

# compare all registered model versions
make compare

# promote the best version to production
make promote VERSION=10 ALIAS=production

# serve the production model (no run_id needed!)
make serve MODEL_URI="models:/afrr-capacity-price-xgboost@production"
```

Supported aliases: `staging`, `production`, `champion`, `challenger`

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
## Data Drift Monitoring
Detects distribution shifts in forecast features using [Evidently](https://www.evidentlyai.com/). Compares a reference dataset (training data) against current production data.

```bash
# install monitoring dependencies
make install-monitoring

# run drift check (generates HTML report + JSON summary)
make drift-check REFERENCE_DATA=data/sample.csv CURRENT_DATA=data/current.csv

# CI mode — exits with code 1 if drift is detected
make drift-check-ci REFERENCE_DATA=data/sample.csv CURRENT_DATA=data/current.csv
```

**Monitored features:** FCR price, consumption/gas/spot/solar/wind forecasts, rolling stats (7-day mean/min/max), calendar (holiday/weekend).

**Statistical tests supported:** `ks` (default), `wasserstein`, `psi`, `kl_div`, `jensen_shannon`.

Reports are saved to `reports/` (HTML for visual inspection, JSON for CI/alerting).

## What I'd add next
- Hyperparameter tuning with Bayesian optimization (BayesSearchCV + TimeSeriesSplit)
- Train all 8 models per hour (2 targets × 4 horizons) in a single pipeline run
- Model promotion flow (Staging → Production) in MLflow before deploying
- Backtesting framework to evaluate rolling-window performance over time
- The EventBridge schedule wired to a real retrain trigger
