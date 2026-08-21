from l2_baseline.harness import _extract_evidence
from l2_baseline.models import CitableItem, CitationSelection, Evidence, RetrievalResult


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
