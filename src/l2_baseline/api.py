import asyncio
import json
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from openai import APIConnectionError, APIStatusError, APITimeoutError

from .harness import L2Harness
from .models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatRequest,
    ChatResponse,
    ModelCard,
    ModelList,
    OpenAIResponseMessage,
    dump_messages,
)

SERVED_MODEL = "Lunit/L2-preview"
MODE = "bounded-source-retrieval"
MODEL_CONCURRENCY = 8
_model_slot = asyncio.Semaphore(MODEL_CONCURRENCY)


class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"


app = FastAPI(
    title="Lunit L2 Medical Chatbot Driver",
    version="1.0.0",
    default_response_class=UTF8JSONResponse,
)
_harness: L2Harness | None = None


def _log(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, **fields}), flush=True)


def get_harness() -> L2Harness:
    global _harness
    if _harness is None:
        _harness = L2Harness()
    return _harness


@app.get("/health")
async def health() -> dict[str, str | int]:
    return {"status": "ok", "mode": MODE, "model_concurrency": MODEL_CONCURRENCY}


@app.get("/v1/models", response_model=ModelList)
async def list_models() -> ModelList:
    return ModelList(data=[ModelCard(id=SERVED_MODEL)])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse | StreamingResponse:
    request_id = uuid.uuid4().hex
    messages = dump_messages(request.messages)
    started = time.monotonic()
    _log(
        "request_started",
        request_id=request_id,
        message_count=len(messages),
        stream=request.stream,
    )
    try:
        harness = get_harness()
        async with _model_slot:
            async with asyncio.timeout(harness.settings.turn_timeout_sec):
                answer = await harness.chat(messages)
    except ValueError as exc:
        _log("request_failed", request_id=request_id, kind="input")
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except TimeoutError as exc:
        _log("request_failed", request_id=request_id, kind="turn_timeout")
        raise HTTPException(status_code=504, detail="Turn exceeded the time limit") from exc
    except APITimeoutError as exc:
        _log("request_failed", request_id=request_id, kind="upstream_timeout")
        raise HTTPException(status_code=504, detail="Lunit L2 upstream timed out") from exc
    except APIStatusError as exc:
        _log(
            "request_failed",
            request_id=request_id,
            kind="upstream_http",
            status_code=exc.status_code,
        )
        raise HTTPException(status_code=502, detail="Lunit L2 upstream unavailable") from exc
    except APIConnectionError as exc:
        _log("request_failed", request_id=request_id, kind="upstream_connection")
        raise HTTPException(status_code=502, detail="Lunit L2 upstream unavailable") from exc
    except RuntimeError as exc:
        _log("request_failed", request_id=request_id, kind="invalid_response")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    completion = ChatCompletionResponse(
        id=f"chatcmpl-{request_id}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(message=OpenAIResponseMessage(content=answer))
        ],
    )
    _log(
        "request_succeeded",
        request_id=request_id,
        elapsed_ms=round((time.monotonic() - started) * 1_000),
        answer_chars=len(answer),
    )
    if request.stream:
        return StreamingResponse(
            iter([_event_stream_payload(completion)]),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )
    return completion


def _event_stream_payload(completion: ChatCompletionResponse) -> str:
    choice = completion.choices[0]
    common = {
        "id": completion.id,
        "object": "chat.completion.chunk",
        "created": completion.created,
        "model": completion.model,
    }
    chunks = [
        {
            **common,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": choice.message.content},
                    "finish_reason": None,
                }
            ],
        },
        {
            **common,
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": choice.finish_reason}
            ],
        },
    ]
    events = [f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n" for chunk in chunks]
    events.append("data: [DONE]\n\n")
    return "".join(events)


# Small development endpoint retained for local clients created before submission formatting.
@app.post("/chat", response_model=ChatResponse, include_in_schema=False)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        harness = get_harness()
        async with _model_slot:
            async with asyncio.timeout(harness.settings.turn_timeout_sec):
                answer = await harness.chat(request.messages)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Turn exceeded the time limit") from exc
    return ChatResponse(answer=answer)
