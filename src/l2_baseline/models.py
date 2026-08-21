from typing import Any, Literal, TypeAlias, get_args

from pydantic import BaseModel, Field


class CitableItem(BaseModel):
    cite_uid: str
    relevance_score: float = Field(ge=0, le=1)


class CitationSelection(BaseModel):
    status: Literal["sufficient", "partial", "no_evidence"]
    items: list[CitableItem] = Field(default_factory=list)
    note: str = ""


class PlanDecision(BaseModel):
    requires_retrieval: bool
    self_contained_query: str
    reason: str = ""


RetrievalPipeline: TypeAlias = Literal[
    "direct",
    "index",
    "law",
    "rag_sql",
    "rag_vector",
    "drug_label",
    "drug_substitution",
    "kcd_billing",
]
RETRIEVAL_PIPELINES = tuple(get_args(RetrievalPipeline))


class RetrievalQueryDecision(BaseModel):
    query_sufficient: bool
    pipeline_type: RetrievalPipeline = "direct"
    reason: str = ""


class Evidence(BaseModel):
    cite_uid: str
    relevance_score: float
    content: str
    bm25_score: float = 0


class RetrievalResult(BaseModel):
    status: Literal["sufficient", "partial", "no_evidence"]
    evidence: list[Evidence] = Field(default_factory=list)
    note: str = ""
    tool_calls: int = 0

    def for_generation(self) -> str:
        lines = [f"status: {self.status}"]
        if self.note:
            lines.append(f"note: {self.note}")
        for index, item in enumerate(self.evidence, 1):
            lines.extend(
                [
                    "",
                    f"[{index}]",
                    f"cite_uid: {item.cite_uid}",
                    f"relevance_score: {item.relevance_score:.2f}",
                    item.content,
                ]
            )
        if not self.evidence:
            lines.append("No citable evidence was selected.")
        return "\n".join(lines)


class ChatRequest(BaseModel):
    messages: list[dict[str, str]]


class ChatResponse(BaseModel):
    answer: str


class OpenAIMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[OpenAIMessage]
    stream: bool = False


class OpenAIResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: OpenAIResponseMessage
    finish_reason: Literal["stop"] = "stop"


class CompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: CompletionUsage = Field(default_factory=CompletionUsage)


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "lunit-hackathon-submission"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


def dump_messages(messages: list[OpenAIMessage]) -> list[dict[str, Any]]:
    return [message.model_dump() for message in messages]
