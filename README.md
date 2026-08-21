# Lunit L2 Contract-Aware Retrieval Driver

Lunit Hackathon 제출 규격을 따르는 containerized multi-turn conversation driver입니다.
최종 답변은 `Lunit/L2-preview`가 생성합니다.

## 처리 흐름

1. 첫 user 질문과 최근 5개 message를 유지하고 message별 길이를 제한합니다.
2. L2가 답하기 전에 최신 요청의 task/artifact, audience, format, length, 제외 항목,
   missing context, jurisdiction, evidence need를 answer contract로 적용합니다.
3. 문서작성·교정·template 요청은 retrieval보다 요청된 산출물 생성을 우선합니다.
4. 일반 의료 질문은 direct로 답합니다. 용량·간격·조절처럼 정밀한 임상 질문은 PubMed
   vector retrieval로 좁혀 무관한 drug-label tool 선택을 막고, 최신 guideline·법령·허가·연구
   요청은 해당 공식 source retrieval로 보냅니다.
5. Guideline은 document → relevant section → page의 최대 3-step index 조회를 사용합니다.
   그 밖의 근거 질문에서는 L2가 학습한 MCP tool 21개 전체 중 하나를 선택합니다.
6. 저자원 환경과 핵심 자료가 누락된 위험평가는 전용 guarded generation으로 처리하고,
   침습적 자가 처치·구체적 자가 투약·근거 없는 저위험 판단이 나오면 한 번 재생성합니다.
   단일 수치와 관할권 확인도 각각 전용 schema 한 개를 `tool_choice=required`로 강제합니다.
7. 1차 경로의 deadline을 넘기면 남겨 둔 예산으로 L2 direct fallback을 실행합니다.

동시에 최대 8개 turn을 처리하며, 대기열에서 기다린 시간은 실제 turn의 120초 처리 제한에
포함하지 않습니다.

반복적인 HyDE/reflection loop는 사용하지 않습니다. Retrieval이 partial/no-evidence이거나
제한 시간을 넘겨도 최종 L2 generation은 실제 질문에 답하고, 확인되지 않은 source-specific
부분만 짧게 구분합니다.

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
L2_REQUEST_TIMEOUT_SEC=50
L2_TURN_TIMEOUT_SEC=120
L2_FALLBACK_RESERVE_SEC=36
L2_VERIFIER_TIMEOUT_SEC=30
L2_RETRIEVAL_TIMEOUT_SEC=28
L2_MAX_RETRIEVAL_CALLS=3
L2_MAX_TOOL_RESULT_CHARS=6000
L2_MAX_EVIDENCE_CHARS=10000
L2_RETRIEVAL_MAX_TOKENS=512
L2_GENERATION_MAX_TOKENS=2048
L2_CONCISE_MAX_TOKENS=768
L2_VERIFICATION_MAX_TOKENS=2048
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
