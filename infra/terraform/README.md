# Terraform Scaffold

This directory contains a minimal Terraform scaffold for:

- SageMaker model + endpoint config + endpoint
- Optional EventBridge Scheduler trigger for a SageMaker Pipeline
- CloudWatch alarms for pipeline failures and endpoint 5XX responses

## Quick Validation

```bash
terraform -chdir=infra/terraform init -backend=false
terraform -chdir=infra/terraform validate
```

## Plan (after filling variables)

```bash
cp infra/terraform/terraform.tfvars.example infra/terraform/terraform.tfvars
terraform -chdir=infra/terraform plan -var-file=terraform.tfvars
```

