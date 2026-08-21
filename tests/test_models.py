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

    rendered = result.for_generation(max_chars=2_000)
    assert 1_800 <= len(rendered) <= 2_000


def test_generation_context_balances_multiple_evidence_items() -> None:
    result = RetrievalResult(
        status="partial",
        evidence=[
            Evidence(cite_uid="cite-window", relevance_score=0.9, content="w" * 20_000),
            Evidence(
                cite_uid="cite-contraindications",
                relevance_score=0.85,
                content="contraindication details " + "c" * 20_000,
            ),
        ],
    )

    rendered = result.for_generation(max_chars=2_000)

    assert 1_800 <= len(rendered) <= 2_000
    assert "[1]" in rendered
    assert "[2]" in rendered
    assert "cite-window" in rendered
    assert "cite-contraindications" in rendered
    assert "contraindication details" in rendered
