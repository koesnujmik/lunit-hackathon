FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml README.md .env.example ./
COPY src ./src

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["python", "-m", "lunit_hackathon.server"]
