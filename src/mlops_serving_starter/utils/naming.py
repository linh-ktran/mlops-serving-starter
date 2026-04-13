from __future__ import annotations

import re

MAX_SAGEMAKER_NAME_LENGTH = 63
DEFAULT_RESOURCE_NAME = "mlops-serving-starter"


# Keep SageMaker-compatible names in one place so deploy and pipeline stay consistent.
def sanitize_resource_name(value: str, *, max_length: int = MAX_SAGEMAKER_NAME_LENGTH) -> str:
    normalized = re.sub(r"[^A-Za-z0-9-]+", "-", value).strip("-")
    normalized = re.sub(r"-+", "-", normalized)
    if not normalized:
        normalized = DEFAULT_RESOURCE_NAME
    return normalized[:max_length]

