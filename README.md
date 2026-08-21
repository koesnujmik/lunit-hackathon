# Lunit L2 Direct Trial Driver

Lunit Hackathon 제출 규격을 따르는 containerized multi-turn conversation driver입니다.
최종 답변은 `Lunit/L2-preview`가 생성합니다.

## 처리 흐름

1. 최근 4개 message만 유지하고 message별 길이를 제한합니다.
2. MCP retrieval과 tool call 없이 L2를 정확히 한 번 호출합니다.
3. 전체 turn을 55초로 제한하고 L2 SDK retry를 끕니다.
4. 일반 JSON과 OpenAI-compatible SSE streaming 응답을 모두 지원합니다.

이 버전은 장시간 retrieval로 CoEval 전체가 중단되는 문제를 분리하기 위한 안정성 baseline입니다.
bounded retrieval은 direct-only trial 완주를 확인한 뒤 tool 1~2회 제한으로 다시 추가합니다.

## 로컬 Python 실행

Python 3.11 이상이 필요합니다.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env
uvicorn l2_baseline.api:app --host 0.0.0.0 --port 8000
```

로컬에서는 `.env`의 `LUNIT_FM_API_KEY`를 사용합니다. CoEval은 runtime secret을 주입하지
않으므로 제출 image는 repository root의 `submission_api_key`를 fallback으로 사용합니다.
런타임 `LUNIT_FM_API_KEY`, `OPENAI_API_KEY`, `LUNIT_API_KEY`가 있으면 파일보다 우선합니다.
URL, model, timeout 설정은 코드 기본값이 있어 별도로 hardcode할 필요가 없습니다.

## 제출 API

- `GET /v1/models`
- `POST /v1/chat/completions`
- `GET /health`

```powershell
$body = @{
  model = "Lunit/L2-preview"
  messages = @(
    @{ role = "user"; content = "만성 신장질환 환자의 권고 혈압 목표는?" }
  )
} | ConvertTo-Json -Depth 10

Invoke-RestMethod `
  http://127.0.0.1:8000/v1/chat/completions `
  -Method Post `
  -ContentType application/json `
  -Body $body
```

Evaluator가 전달한 `messages` 중 최근 4개를 별도의 session ID 없이 사용합니다.
`stream: true`에는 OpenAI-compatible SSE 형식으로 응답합니다.

## Docker 검증

```powershell
docker build -t lunit-submission:local .
docker run --rm -p 8000:8000 lunit-submission:local
```

Container는 `.env` 없이 image 내부의 `submission_api_key`를 읽고 `0.0.0.0:8000`에서
시작하며 Dockerfile은 `EXPOSE 8000`을 선언합니다.

## 설정

```text
L2_REQUEST_TIMEOUT_SEC=45
L2_TURN_TIMEOUT_SEC=55
L2_GENERATION_MAX_TOKENS=1536
L2_MAX_HISTORY_MESSAGES=4
L2_MAX_MESSAGE_CHARS=6000
LUNIT_SUBMISSION_API_KEY_FILE=submission_api_key
```

## 테스트

```powershell
pytest
ruff check .
```

## 제출

```powershell
git switch lunit/hackathon-submission
git add .
git commit -m "Format containerized OpenAI-compatible submission driver"
git push origin lunit/hackathon-submission
git rev-parse HEAD
```

마지막 명령의 40자리 SHA와 model 이름 `Lunit/L2-preview`를 제출 페이지에 입력합니다.
