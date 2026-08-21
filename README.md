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
입력 길이 문제로 실패할 때만 최근 4개 메시지로 한 번 재시도합니다. 인증 오류처럼 재시도로
해결되지 않는 오류는 즉시 종료합니다. 정상 assistant 응답은 항상 Lunit L2가 생성하며, L2가
최종적으로 실패하면 고정 문장을 대신 반환하지 않고 OpenAI-compatible error response를 보냅니다.

시작 로그의 `driver_starting` 이벤트에는 API key 값이 아닌 존재 여부와 Model API host만
기록됩니다. 요청 실패 시 `request_failed` 이벤트에서 오류 종류와 upstream HTTP status를
확인할 수 있습니다.

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
