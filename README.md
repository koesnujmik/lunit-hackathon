# Lunit L2 Native MCP Two-Stage Driver

Lunit Hackathon 제출 규격을 따르는 containerized multi-turn conversation driver입니다.
최종 답변은 `Lunit/L2-preview`가 생성합니다.

## 처리 흐름

1. Evaluator가 보낸 전체 conversation history와 `retrieve_relevant_content` bridge tool 하나를
   generation 단계의 L2에 전달합니다.
2. L2가 memory로 답할 수 있으면 MCP를 열지 않고 첫 generation 응답을 바로 반환합니다.
3. 근거가 필요해 L2가 bridge tool을 호출한 경우에만 retrieval 단계를 시작합니다.
4. Lunit MCP server에서 tool schema를 실시간으로 받아 실제 tool 21개 전체를 retrieval L2에
   제공합니다.
5. L2가 native tool call을 선택하면 harness가 MCP를 호출하고, assistant tool call과 tool 결과를
   같은 retrieval conversation에 이어 붙입니다.
6. 최대 3라운드·4회 범위에서 필요한 후속 MCP tool을 호출합니다. 문서 탐색처럼 선행 tool이
   필요한 workflow도 이 연속된 trajectory 안에서 처리합니다.
7. `cite_uid`가 있는 실제 MCP 결과를 bridge tool 결과로 generation trajectory에 돌려보내고
   L2가 최종 assistant response를 만듭니다.

HyDE와 tool schema 사전 필터링은 사용하지 않습니다. `retrieve_relevant_content`는 실제 MCP
tool이 아니라 공식 권장 구조에 따라 두 단계를 연결하는 local orchestration tool입니다.
Retrieval이 제한 시간 내 실패해도 service가 죽지 않고 evidence 없음 상태로 generation을 계속합니다.

## 로컬 Python 실행

Python 3.11 이상이 필요합니다.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env
uvicorn l2_baseline.api:app --host 0.0.0.0 --port 8000
```

필요하면 `.env`의 `LUNIT_FM_API_KEY`로 제출용 설정을 덮어쓸 수 있습니다.

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

Evaluator가 전달한 전체 `messages` history를 별도의 session ID 없이 매 요청의 대화 문맥으로
사용합니다. Streaming 요청은 지원하지 않습니다.

## Docker 검증

```powershell
docker build -t lunit-submission:local .
docker run --rm -p 8000:8000 lunit-submission:local
```

환경변수로 설정을 덮어쓸 때만 `--env-file .env`를 추가합니다. Docker env 파일은
`export KEY=value`가 아니라 `KEY=value` 형식이어야 합니다.

Container는 별도 작업 없이 `0.0.0.0:8000`에서 시작하며 Dockerfile은 `EXPOSE 8000`을
선언합니다.

## 설정

```text
L2_MAX_RETRIEVAL_CALLS=4
L2_MAX_RETRIEVAL_ROUNDS=3
L2_RETRIEVAL_TOP_K=3
L2_REQUEST_TIMEOUT_SEC=45
L2_RETRIEVAL_TIMEOUT_SEC=45
L2_MAX_TOOL_RESULT_CHARS=8000
L2_MAX_EVIDENCE_CHARS=16000
L2_RETRIEVAL_MAX_TOKENS=768
L2_GENERATION_MAX_TOKENS=6144
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
