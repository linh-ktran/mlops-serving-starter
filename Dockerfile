FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY configs ./configs
COPY pyproject.toml README.md ./

ENV PYTHONPATH=/app/src

EXPOSE 8000
CMD ["python", "-m", "mlops_serving_starter.api.app"]
