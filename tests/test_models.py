import pytest
from pydantic import ValidationError

from l2_baseline.harness import QUERY_ASSESSMENT_TOOL, _extract_evidence
from l2_baseline.models import (
    RETRIEVAL_PIPELINES,
    CitableItem,
    CitationSelection,
    Evidence,
    RetrievalQueryDecision,
    RetrievalResult,
)
from l2_baseline.pipelines import select_pipeline_type


def test_selected_citation_is_forwarded() -> None:
    selection = CitationSelection(
        status="sufficient",
        items=[CitableItem(cite_uid="cite-123", relevance_score=0.9)],
    )
    evidence = _extract_evidence(["unrelated", '{"cite_uid":"cite-123","content":"fact"}'], selection)
    assert len(evidence) == 1
    assert "fact" in evidence[0].content


def test_generation_context_has_numbered_citations() -> None:
    result = RetrievalResult(
        status="partial",
        evidence=[Evidence(cite_uid="cite-1", relevance_score=0.8, content="source text")],
    )
    rendered = result.for_generation()
    assert "status: partial" in rendered
    assert "[1]" in rendered
    assert "source text" in rendered


def test_pipeline_type_is_deterministic_and_model_validated() -> None:
    properties = QUERY_ASSESSMENT_TOOL["function"]["parameters"]["properties"]

    assert "pipeline_type" not in properties
    assert select_pipeline_type("According to clinical guidelines, what is the target?") == "index"
    assert select_pipeline_type("국민건강보험법 시행령의 해당 조문은?") == "law"
    assert select_pipeline_type("FAERS reporting odds ratio for warfarin") == "rag_sql"
    assert select_pipeline_type("Official label interactions for warfarin") == "direct"
    assert set(RETRIEVAL_PIPELINES) == {
        "direct",
        "index",
        "law",
        "rag_sql",
        "rag_vector",
        "drug_label",
        "drug_substitution",
        "kcd_billing",
    }
    assert RetrievalQueryDecision(
        query_sufficient=True, pipeline_type="law"
    ).pipeline_type == "law"
    with pytest.raises(ValidationError):
        RetrievalQueryDecision(query_sufficient=True, pipeline_type="unknown")
