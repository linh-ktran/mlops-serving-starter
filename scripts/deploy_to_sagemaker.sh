#!/bin/bash
# =============================================================================
# Deploy model to SageMaker (end-to-end)
#
# Prerequisites:
#   - AWS CLI configured (aws configure / SSO)
#   - Docker installed
#   - A trained model (run make train first)
#
# Usage:
#   ./scripts/deploy_to_sagemaker.sh <RUN_ID> <AWS_ACCOUNT_ID> <REGION>
#
# Example:
#   ./scripts/deploy_to_sagemaker.sh 7ced553ee90f41a1a2a13a78b83da180 123456789012 eu-west-1
#
# To clean up after testing:
#   ./scripts/deploy_to_sagemaker.sh --cleanup <AWS_ACCOUNT_ID> <REGION>
# =============================================================================
set -euo pipefail

ENDPOINT_NAME="mlops-serving-starter-endpoint"
ECR_REPO_NAME="mlops-serving-starter"
INSTANCE_TYPE="ml.t2.medium"  # cheapest option (~€0.065/hour)
S3_BUCKET="${S3_BUCKET:-your-bucket-name}"
S3_PREFIX="${S3_PREFIX:-mlflow/mlops-serving-starter}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# --- Cleanup mode ---
if [ "${1:-}" = "--cleanup" ]; then
    ACCOUNT_ID="${2:?Usage: $0 --cleanup <ACCOUNT_ID> <REGION>}"
    REGION="${3:?Usage: $0 --cleanup <ACCOUNT_ID> <REGION>}"

    info "Deleting SageMaker endpoint..."
    aws sagemaker delete-endpoint --endpoint-name "$ENDPOINT_NAME" --region "$REGION" 2>/dev/null || true

    info "Deleting endpoint config..."
    aws sagemaker delete-endpoint-config --endpoint-config-name "${ENDPOINT_NAME}-config" --region "$REGION" 2>/dev/null || true

    info "Deleting model..."
    aws sagemaker delete-model --model-name "${ENDPOINT_NAME}-model" --region "$REGION" 2>/dev/null || true

    info "✓ All resources deleted. No more charges."
    exit 0
fi

# --- Deploy mode ---
RUN_ID="${1:?Usage: $0 <RUN_ID> <ACCOUNT_ID> <REGION>}"
ACCOUNT_ID="${2:?Usage: $0 <RUN_ID> <ACCOUNT_ID> <REGION>}"
REGION="${3:?Usage: $0 <RUN_ID> <ACCOUNT_ID> <REGION>}"

# Resolve the actual AWS account from the active profile/credentials
ACTUAL_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
info "Active AWS account: $ACTUAL_ACCOUNT_ID (profile: ${AWS_PROFILE:-default})"

ECR_URI="${ACTUAL_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO_NAME}:latest"
MODEL_URI="runs:/${RUN_ID}/model"
MODEL_S3_PATH="s3://${S3_BUCKET}/${S3_PREFIX}/model.tar.gz"

echo ""
info "=== SageMaker Deployment Plan ==="
echo "  Run ID:        $RUN_ID"
echo "  Region:        $REGION"
echo "  ECR Image:     $ECR_URI"
echo "  Model S3:      $MODEL_S3_PATH"
echo "  Instance:      $INSTANCE_TYPE"
echo "  Endpoint:      $ENDPOINT_NAME"
echo ""
echo "  Estimated cost: ~€0.065/hour (delete after testing!)"
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo
[[ $REPLY =~ ^[Yy]$ ]] || exit 0

# Step 1: Package the model
info "Step 1/5: Packaging model..."
MLFLOW_TRACKING_URI="file://$(pwd)/mlruns" .venv/bin/python scripts/package_model_for_sagemaker.py \
    --model-uri "$MODEL_URI" \
    --output artifacts/model.tar.gz

# Step 2: Upload model to S3
info "Step 2/5: Uploading model to S3..."
aws s3 cp artifacts/model.tar.gz "$MODEL_S3_PATH" --region "$REGION" --sse AES256

# Step 3: Build and push Docker image to ECR
info "Step 3/5: Building and pushing Docker image..."

# Create ECR repo if it doesn't exist
aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$REGION" 2>/dev/null || \
    aws ecr create-repository --repository-name "$ECR_REPO_NAME" --region "$REGION"

# Login to ECR
aws ecr get-login-password --region "$REGION" | \
    docker login --username AWS --password-stdin "${ACTUAL_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# Build
docker build -t "$ECR_REPO_NAME" -f docker/Dockerfile.sagemaker .

# Tag and push
docker tag "${ECR_REPO_NAME}:latest" "$ECR_URI"
docker push "$ECR_URI"

# Step 4: Get SageMaker execution role
info "Step 4/5: Looking up SageMaker execution role..."
ROLE_ARN=$(aws iam list-roles --output json | python3 -c "
import sys, json
roles = json.load(sys.stdin)['Roles']
for r in roles:
    name = r['RoleName']
    if 'sagemaker' in name.lower() and 'AWSServiceRole' not in name:
        print(r['Arn'])
        break
")
if [ -z "$ROLE_ARN" ]; then
    error "No SageMaker IAM role found. Create one with AmazonSageMakerFullAccess policy or set ROLE_ARN manually."
fi
info "Using role: $ROLE_ARN"

# Step 5: Deploy endpoint
info "Step 5/5: Deploying to SageMaker..."
MLFLOW_TRACKING_URI="file://$(pwd)/mlruns" .venv/bin/python scripts/sagemaker_deploy_example.py \
    --apply \
    --endpoint-name "$ENDPOINT_NAME" \
    --image-uri "$ECR_URI" \
    --model-data-url "$MODEL_S3_PATH" \
    --execution-role-arn "$ROLE_ARN" \
    --instance-type "$INSTANCE_TYPE" \
    --region "$REGION"

echo ""
info "=== Deployment started ==="
info "Endpoint will be ready in ~5-8 minutes."
info ""
info "Check status:"
info "  aws sagemaker describe-endpoint --endpoint-name $ENDPOINT_NAME --region $REGION --query 'EndpointStatus'"
info ""
info "Test it (once InService):"
info "  aws sagemaker-runtime invoke-endpoint \\"
info "    --endpoint-name $ENDPOINT_NAME \\"
info "    --region $REGION \\"
info "    --content-type application/json \\"
info "    --body '{\"records\":[{\"afrr_capacity_price_up_lag_1_h1\":25.3,\"afrr_capacity_price_up_lag_2_h1\":24.1,\"afrr_capacity_price_up_lag_3_h1\":23.5,\"afrr_capacity_price_up_lag_7_h1\":22.0,\"afrr_capacity_price_up_lag_14_h1\":21.5,\"afrr_capacity_price_up_lag_28_h1\":20.0,\"rolling_mean_7\":23.4,\"rolling_min_7\":18.2,\"rolling_max_7\":30.1,\"fcr_price_symmetric\":15.0,\"consumption_forecast\":52000.0,\"gas_price_forecast\":32.5,\"spot_price_forecast\":55.0,\"solar_forecast\":2500.0,\"wind_onshore_forecast\":2100.0,\"wind_offshore_forecast\":900.0,\"holiday_status\":0,\"weekend_status\":0}]}' \\"
info "    /dev/stdout"
info ""
warn "⚠️  REMEMBER TO DELETE AFTER TESTING:"
warn "  ./scripts/deploy_to_sagemaker.sh --cleanup $ACCOUNT_ID $REGION"


