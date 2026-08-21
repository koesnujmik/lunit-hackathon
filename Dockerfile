FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

COPY pyproject.toml README.md ./
COPY submission_api_key ./submission_api_key
COPY src ./src
RUN test -s /app/submission_api_key
RUN pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "l2_baseline.api:app", "--host", "0.0.0.0", "--port", "8000"]
