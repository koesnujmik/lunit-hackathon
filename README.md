# Lunit L2 Multi-Agent RAG Driver

Lunit Hackathon 제출 규격을 따르는 containerized multi-turn conversation driver입니다.
최종 답변은 `Lunit/L2-preview`가 생성합니다.

## 처리 흐름

1. Router가 전체 대화 문맥을 보고 memory 답변과 retrieval 필요 여부를 결정합니다.
2. Retrieval이 필요하면 agent가 멀티턴 문맥이 해결된 query만으로 tool argument 생성과
   결과 ranking이 충분한지 `query_sufficient`로 판단합니다. 별도의 결정론적 router가
   명시된 source·질문 유형으로 `pipeline_type`을 확정합니다.
3. `query_sufficient=false`일 때만 L2가 필요한 근거·용어·출처 조건을 정리한
   짧은 retrieval rationale를 한 번 생성합니다. Rationale는 검색 보조 정보일 뿐
   evidence로 인용하지 않습니다.
4. `pipeline_type=direct`면 query와 선택적 rationale로 MCP tool schema를 BM25 정렬해
   Top-5만 selector에 제공합니다. Selector는 최대 5개 action을 고르고, 선택된
   action은 단일 batch로 병렬 실행됩니다.
5. Specialized pipeline은 `index`, `law`, `rag_sql`, `rag_vector`, `drug_label`,
   `drug_substitution`, `kcd_billing`입니다. 각 pipeline은 미리 정해진 bounded stage만
   실행하며 전체 MCP 호출은 최대 5번입니다.
6. 모든 pipeline 결과는 query-only 또는 query+rationale BM25로 정렬하고,
   `cite_uid`가 있는 context Top-2만 유지합니다.
7. `finalize_retrieval`이 최종 citation과 근거 충분성을 확정하고, L2가 선택된
   context로 최종 답변을 생성합니다.

이 retrieval은 ReAct나 Reflection 반복 루프를 사용하지 않습니다. Direct와 specialized
pipeline 모두 stage와 MCP 호출 수가 고정 상한 안에서 종료됩니다.

| `pipeline_type` | 고정 실행 경로 |
|---|---|
| `direct` | BM25 schema Top-5 → selector → 단일 병렬 batch |
| `index` | relevant-node/keyword 검색 → 반환된 `doc_id`·page range의 원문 조회 |
| `law` | 법령명 검색 → `mst` 조문 목록 → 선택한 `article_keys` 전문 조회 |
| `rag_sql` | 고정 SQL source schema → schema 기반 SQL 조회 |
| `rag_vector` | 고정 vector source metadata → semantic 조회 |
| `drug_label` | MFDS 한국 제품 → `ingredient_eng` → DailyMed label |
| `drug_substitution` | MFDS 성분 확인 → 동일성분 제품 → 후보 적응증 |
| `kcd_billing` | 질병명 code 검색 → 공식 명칭·HIRA 청구 유효성 병렬 확인 |

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
