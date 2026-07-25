FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN pip install --no-cache-dir .

ENV MODEL_PATH=/app/artifacts/hybrid_model.pt
EXPOSE 8000
CMD ["uvicorn", "recsys.api:app", "--host", "0.0.0.0", "--port", "8000"]
