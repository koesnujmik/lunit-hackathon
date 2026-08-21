# Lunit L2 Bounded Retrieval Driver

Lunit Hackathon 제출 규격을 따르는 containerized multi-turn conversation driver입니다.
최종 답변은 `Lunit/L2-preview`가 생성합니다.

## 처리 흐름

1. 첫 user 질문과 최근 5개 message를 유지하고 message별 길이를 제한합니다.
2. Generation L2에는 `retrieve_relevant_content` 하나만 제공하며, L2가 memory로 직접
   답할지 self-contained query로 retrieval을 요청할지 결정합니다.
3. Retrieval 결과는 `role=tool` message로 같은 generation 대화에 전달하고 최종 답변을
   생성합니다.
4. Retrieval L2에는 실제 MCP tools와 `finalize_retrieval`을 함께 제공합니다. Guideline은
   결정적 2-step index 조회를 사용하고, 그 밖의 근거 질문은 L2가 MCP action을 선택합니다.
5. MCP tool은 최대 2회 호출한 뒤 L2가 `finalize_retrieval`로 `status`, `note`, 관련
   `cite_uid`를 제출하며, 실제 tool 결과에 존재하는 citation 본문만 generation에 전달합니다.
6. 전체 turn을 55초로 제한하고 일반 JSON과 OpenAI-compatible SSE를 지원합니다.

동시에 최대 8개 turn을 처리하며, 대기열에서 기다린 시간은 실제 turn의 55초 처리 제한에
포함하지 않습니다.

반복적인 HyDE/reflection loop는 사용하지 않습니다. Retrieval이 실패하거나 제한 시간을 넘겨도
최종 L2 generation은 실행해 안전한 일반 답변과 근거 한계를 전달합니다.

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

Evaluator가 전달한 `messages` 중 첫 user 질문과 최근 5개를 별도의 session ID 없이 사용합니다.
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
L2_RETRIEVAL_TIMEOUT_SEC=18
L2_MAX_RETRIEVAL_CALLS=2
L2_MAX_TOOL_RESULT_CHARS=6000
L2_MAX_EVIDENCE_CHARS=10000
L2_RETRIEVAL_MAX_TOKENS=512
L2_GENERATION_MAX_TOKENS=2048
L2_MAX_HISTORY_MESSAGES=6
L2_MAX_MESSAGE_CHARS=4000
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
