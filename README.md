# Lunit L2 Multi-Agent RAG Driver

Lunit Hackathon 제출 규격을 따르는 containerized multi-turn conversation driver입니다.
최종 답변은 `Lunit/L2-preview`가 생성합니다.

## 처리 흐름

1. Router가 전체 대화 문맥을 보고 memory 답변과 retrieval 필요 여부를 결정합니다.
2. Retrieval이 필요하면 멀티턴 문맥이 해결된 query가 그 자체로 tool 선택과
   ranking에 충분한지 판단합니다.
3. Query가 부족한 경우에만 L2가 필요한 근거·용어·출처 조건을 정리한 짧은
   retrieval rationale를 생성합니다. Rationale는 검색 보조 정보일 뿐 evidence로
   인용하지 않습니다.
4. Query와 선택적 rationale를 기준으로 MCP tool schema를 BM25 정렬하고 Top-5만
   selector에 제공합니다.
5. Selector가 직접 유용한 action을 최대 5개 선택하면, 한 번의 batch로 병렬
   실행합니다.
6. Query가 충분했다면 query-only BM25를, rationale가 필요했다면 query+rationale
   BM25를 사용해 `cite_uid`가 있는 MCP context Top-2를 선정합니다.
7. `finalize_retrieval`이 최종 citation과 근거 충분성을 확정하고, L2가 선택된 context로
   최종 답변을 생성합니다.

MCP tool schema는 프로세스 수명 동안 캐싱하여 첫 retrieval 이후에는 `list_tools`
네트워크 호출을 반복하지 않습니다.

## 로컬 Python 실행

Python 3.11 이상이 필요합니다.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env
uvicorn l2_baseline.api:app --host 0.0.0.0 --port 8000
```

`.env`에서 `LUNIT_FM_API_KEY`를 설정합니다. `.env`와 API key는 commit하지 않습니다.

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
docker run --rm -p 8000:8000 --env-file .env lunit-submission:local
```

Container는 별도 작업 없이 `0.0.0.0:8000`에서 시작하며 Dockerfile은 `EXPOSE 8000`을
선언합니다.

## 설정

```text
L2_MAX_RETRIEVAL_CALLS=5
L2_TOOL_CANDIDATE_LIMIT=5
L2_REQUEST_TIMEOUT_SEC=90
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
