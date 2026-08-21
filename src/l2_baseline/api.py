import time
import uuid

from fastapi import FastAPI, HTTPException

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

app = FastAPI(title="Lunit L2 Medical Chatbot Driver", version="1.0.0")
_harness: L2Harness | None = None


def get_harness() -> L2Harness:
    global _harness
    if _harness is None:
        _harness = L2Harness()
    return _harness


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models", response_model=ModelList)
async def list_models() -> ModelList:
    return ModelList(data=[ModelCard(id=SERVED_MODEL)])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported by this driver")
    messages = dump_messages(request.messages)
    try:
        answer = await get_harness().chat(messages)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(message=OpenAIResponseMessage(content=answer))
        ],
    )


# Small development endpoint retained for local clients created before submission formatting.
@app.post("/chat", response_model=ChatResponse, include_in_schema=False)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        answer = await get_harness().chat(request.messages)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ChatResponse(answer=answer)
