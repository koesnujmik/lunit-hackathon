# Lunit L2 Minimal Driver

HealthBench evaluation 안정성 확인을 위한 최소 OpenAI-compatible driver입니다.

## Pipeline

```text
POST /v1/chat/completions
  -> conversation 정규화
  -> Lunit/L2-preview 1회 호출
  -> OpenAI-compatible response
```

MCP, retrieval, router, planner, reflection agent를 사용하지 않습니다. 첫 호출이 API 오류나
입력 길이 문제로 실패할 때만 최근 4개 메시지로 한 번 재시도합니다. 두 호출이 모두 실패하면
benchmark 전체가 중단되지 않도록 해당 요청에 안전한 fallback 응답을 반환합니다.

## Environment

```bash
export LUNIT_FM_API_URL=https://model.hackathon.lunit.io
export LUNIT_FM_API_KEY=lunit_...
export LUNIT_FM_MODEL=Lunit/L2-preview
export LUNIT_TIMEOUT_SEC=40
export LUNIT_MAX_TOKENS=2048
```

## Local

```bash
source .env
PYTHONPATH=src python -m lunit_hackathon.server
curl http://127.0.0.1:8000/v1/models
```

## Docker

```bash
docker build -t lunit-minimal:local .
docker run --rm -p 8000:8000 --env-file .env lunit-minimal:local
```

The container listens on `0.0.0.0:8000` and implements:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `GET /health`
