from __future__ import annotations

import re

MAX_SAGEMAKER_NAME_LENGTH = 63
DEFAULT_RESOURCE_NAME = "mlops-serving-starter"


def sanitize_resource_name(value: str, *, max_length: int = MAX_SAGEMAKER_NAME_LENGTH) -> str:
    """Normalize a string into a valid SageMaker resource name (alphanumeric + hyphens)."""
    normalized = re.sub(r"[^A-Za-z0-9-]+", "-", value).strip("-")
    normalized = re.sub(r"-+", "-", normalized)
    if not normalized:
        normalized = DEFAULT_RESOURCE_NAME
    return normalized[:max_length]
