from l2_baseline.models import Evidence, RetrievalResult


def test_generation_context_has_numbered_citations() -> None:
    result = RetrievalResult(
        status="partial",
        evidence=[Evidence(cite_uid="cite-1", relevance_score=0.8, content="source text")],
    )
    rendered = result.for_generation()
    assert "status: partial" in rendered
    assert "[1]" in rendered
    assert "source text" in rendered


def test_generation_context_is_size_bounded() -> None:
    result = RetrievalResult(
        status="partial",
        evidence=[Evidence(cite_uid="cite-1", relevance_score=0.8, content="x" * 20_000)],
    )

    assert len(result.for_generation(max_chars=2_000)) == 2_000
