# Lunit L2 Multi-Agent RAG Driver

Lunit Hackathon 제출 규격을 따르는 containerized multi-turn conversation driver입니다.
최종 답변은 `Lunit/L2-preview`가 생성합니다.

## 처리 흐름

1. Planner가 memory 답변과 retrieval 필요 여부를 판단하고 멀티턴 query를 완결합니다.
2. Retrieval이 필요하면 HyDE 단계의 L2가 가상 정답 대신, 질문 해결에 필요한 근거 내용을
   정리한 retrieval-rationale passage를 생성합니다.
3. Tool selector가 관련 MCP tool만 선택해 실행합니다.
4. 실제 citable MCP 결과를 TF-IDF로 정렬해 top 3를 유지합니다.
5. Reflection agent가 충분성을 검사하고 부족하면 ReAct action을 반복합니다.
6. `finalize_retrieval`로 citation을 확정한 뒤 L2가 답변합니다.

HyDE 단계에서 생성하는 rationale은 최대 140단어로, 필요한 임상 개체·환자 조건·중재와
비교군·요청 결과나 정확한 사실·중요한 예외·공식 출처 유형·관할권과 최신성·한영 검색
동의어를 정리합니다. 질문에 대한 답이나 예상 수치를 만들지 않으며 tool 선택과 TF-IDF
ranking에만 사용하고 실제 evidence로 인용하지 않습니다. 비공개 chain-of-thought 대신 근거
부족분과 다음 action을 설명하는 짧은 `analysis_summary`만 전달합니다.

### Tool dependency workflow

의존성이 있는 MCP tool은 현재 workflow state에서 실행 가능한 단계만 selector에 제공합니다.
각 tool 결과에서 `doc_id`, `node_id`, page range, MST, article key, KCD code, 의약품 식별자를
추출하며, 다음 단계의 필수 인자가 준비되면 추가 L2 planning 없이 자동으로 진행합니다.

```text
guideline/HIRA: list_documents → relevant_nodes/structure → page_content
law:            law_search → list_articles → get_article
KCD:            search_codes → get_name
MFDS:           check_permission/find_by_ingredient → get_drug_indication
```

필수 인자를 자동으로 채울 수 없는 경우에만 해당 workflow의 다음 frontier를 L2에 전달합니다.
DailyMed, HIRA update, PubMed, FAERS처럼 독립적으로 실행 가능한 tool은 기존 selector 흐름을
그대로 사용합니다.

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
L2_MAX_RETRIEVAL_CALLS=8
L2_MAX_REFLECTION_ROUNDS=4
L2_RETRIEVAL_TOP_K=3
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
