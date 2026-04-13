variable "aws_region" {
  type        = string
  description = "AWS region for all resources"
}

variable "name_prefix" {
  type        = string
  description = "Prefix for naming resources"
  default     = "mlops-serving-starter"
}

variable "sagemaker_execution_role_arn" {
  type        = string
  description = "IAM role ARN used by SageMaker model and endpoint"
}

variable "model_data_url" {
  type        = string
  description = "S3 URI to model.tar.gz"
}

variable "image_uri" {
  type        = string
  description = "ECR image URI for inference container"
}

variable "endpoint_instance_type" {
  type        = string
  description = "SageMaker endpoint instance type"
  default     = "ml.m5.large"
}

variable "endpoint_initial_instance_count" {
  type        = number
  description = "Number of endpoint instances"
  default     = 1
}

variable "schedule_enabled" {
  type        = bool
  description = "Enable scheduler for SageMaker Pipeline"
  default     = false
}

variable "schedule_expression" {
  type        = string
  description = "EventBridge scheduler expression"
  default     = "cron(0 2 * * ? *)"
}

variable "schedule_timezone" {
  type        = string
  description = "Scheduler timezone"
  default     = "UTC"
}

variable "pipeline_name" {
  type        = string
  description = "SageMaker pipeline name that schedule should trigger"
}

variable "alarm_actions" {
  type        = list(string)
  description = "SNS topics or actions for CloudWatch alarms"
  default     = []
}

variable "ok_actions" {
  type        = list(string)
  description = "SNS topics or actions for CloudWatch alarm recovery"
  default     = []
}

