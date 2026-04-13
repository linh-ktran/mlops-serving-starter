resource "aws_scheduler_schedule" "pipeline_schedule" {
  count = var.schedule_enabled ? 1 : 0

  name                         = local.schedule_name
  schedule_expression          = var.schedule_expression
  schedule_expression_timezone = var.schedule_timezone
  state                        = var.schedule_enabled ? "ENABLED" : "DISABLED"

  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = "arn:aws:sagemaker:${var.aws_region}:${data.aws_caller_identity.current.account_id}:pipeline/${var.pipeline_name}"
    role_arn = aws_iam_role.scheduler_role[0].arn

    sagemaker_pipeline_parameters {
      pipeline_parameter {
        name  = "InputDateTime"
        value = "NA"
      }
    }
  }
}

data "aws_caller_identity" "current" {}

resource "aws_iam_role" "scheduler_role" {
  count = var.schedule_enabled ? 1 : 0

  name               = "${local.schedule_name}-role"
  assume_role_policy = data.aws_iam_policy_document.scheduler_assume_role[0].json
}

data "aws_iam_policy_document" "scheduler_assume_role" {
  count = var.schedule_enabled ? 1 : 0

  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["scheduler.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy" "scheduler_policy" {
  count = var.schedule_enabled ? 1 : 0

  name = "${local.schedule_name}-policy"
  role = aws_iam_role.scheduler_role[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["sagemaker:StartPipelineExecution"]
        Resource = "arn:aws:sagemaker:${var.aws_region}:${data.aws_caller_identity.current.account_id}:pipeline/${var.pipeline_name}"
      }
    ]
  })
}

