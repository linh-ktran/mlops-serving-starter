resource "aws_sagemaker_model" "this" {
  name               = local.model_name
  execution_role_arn = var.sagemaker_execution_role_arn

  primary_container {
    image          = var.image_uri
    model_data_url = var.model_data_url

    environment = {
      SAGEMAKER_PROGRAM = "mlops_serving_starter/sagemaker/inference.py"
    }
  }
}

resource "aws_sagemaker_endpoint_configuration" "this" {
  name = local.endpoint_config_name

  production_variants {
    variant_name           = "AllTraffic"
    model_name             = aws_sagemaker_model.this.name
    initial_instance_count = var.endpoint_initial_instance_count
    instance_type          = var.endpoint_instance_type
    initial_variant_weight = 1
  }
}

resource "aws_sagemaker_endpoint" "this" {
  name                 = local.endpoint_name
  endpoint_config_name = aws_sagemaker_endpoint_configuration.this.name
}

