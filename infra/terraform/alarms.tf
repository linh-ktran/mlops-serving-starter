resource "aws_cloudwatch_metric_alarm" "pipeline_failed_executions" {
  alarm_name          = "${var.name_prefix}-pipeline-failed-executions"
  alarm_description   = "SageMaker pipeline execution failures"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  threshold           = 1
  treat_missing_data  = "ignore"

  namespace   = "AWS/Sagemaker/ModelBuildingPipeline"
  metric_name = "ExecutionFailed"
  statistic   = "Sum"
  period      = 3600

  dimensions = {
    PipelineName = var.pipeline_name
  }

  alarm_actions = var.alarm_actions
  ok_actions    = var.ok_actions
}

resource "aws_cloudwatch_metric_alarm" "endpoint_5xx_errors" {
  alarm_name          = "${var.name_prefix}-endpoint-5xx"
  alarm_description   = "SageMaker endpoint 5XX errors"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  threshold           = 1
  treat_missing_data  = "notBreaching"

  namespace   = "AWS/SageMaker"
  metric_name = "Invocation5XXErrors"
  statistic   = "Sum"
  period      = 300

  dimensions = {
    EndpointName = aws_sagemaker_endpoint.this.name
    VariantName  = "AllTraffic"
  }

  alarm_actions = var.alarm_actions
  ok_actions    = var.ok_actions
}

