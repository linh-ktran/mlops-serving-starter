FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs ./configs

RUN uv sync --frozen --no-dev

ENV PYTHONPATH=/app/src

EXPOSE 8000
CMD ["uv", "run", "python", "-m", "mlops_serving_starter.api.app"]
