# Teknofest 2026 E-Ticaret — Offline Docker

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/data /app/experiments /app/local_model /app/experiments/outputs

COPY src/ /app/src/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
COPY data/ /app/data/

EXPOSE 8000

ENV MLFLOW_TRACKING_URI=sqlite:////app/experiments/mlflow.db
ENV CONFIG_PATH=configs/base_config.yaml
ENV TRANSFORMERS_OFFLINE=1
ENV LOCAL_MODEL_PATH=/app/local_model

CMD ["uvicorn", "src.deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]
