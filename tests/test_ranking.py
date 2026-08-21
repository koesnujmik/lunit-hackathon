from l2_baseline.ranking import rank_documents


def test_bm25_returns_only_relevant_citable_candidates() -> None:
    passage = "만성 신장질환 chronic kidney disease blood pressure target"
    documents = [
        '{"cite_uid":"cite-a","content":"chronic kidney disease blood pressure target"}',
        '{"cite_uid":"cite-b","content":"blood pressure in CKD"}',
        '{"cite_uid":"cite-c","content":"kidney guideline"}',
        '{"cite_uid":"cite-d","content":"unrelated dermatology"}',
        '{"content":"highly similar chronic kidney disease but not citable"}',
    ]
    ranked = rank_documents(passage, passage, documents)
    assert 2 <= len(ranked) <= 5
    assert ranked[0].cite_uid == "cite-a"
    assert ranked[0].bm25_score > 0
    assert all(item.cite_uid != "cite-d" for item in ranked[:2])


def test_bm25_empty_documents() -> None:
    assert rank_documents("query", "rationale", []) == []


def test_bm25_penalizes_irrelevant_document_length() -> None:
    documents = [
        '{"cite_uid":"cite-short","content":"warfarin interaction"}',
        (
            '{"cite_uid":"cite-long","content":"warfarin interaction '
            + "unrelated " * 100
            + '"}'
        ),
    ]

    ranked = rank_documents(
        "warfarin interaction", "warfarin interaction", documents
    )

    assert [item.cite_uid for item in ranked] == ["cite-short", "cite-long"]
    assert ranked[0].bm25_score > ranked[1].bm25_score


def test_hybrid_bm25_uses_query_and_rationale_with_query_priority() -> None:
    documents = [
        '{"cite_uid":"cite-query","content":"warfarin official interaction"}',
        '{"cite_uid":"cite-rationale","content":"aspirin bleeding monitoring"}',
    ]

    ranked = rank_documents(
        "warfarin official interaction",
        "aspirin bleeding monitoring",
        documents,
    )

    assert [item.cite_uid for item in ranked] == ["cite-query", "cite-rationale"]
    assert ranked[0].bm25_score == 0.65
    assert ranked[1].bm25_score == 0.35


def test_adaptive_cutoff_keeps_minimum_two_candidates() -> None:
    documents = [
        '{"cite_uid":"cite-match","content":"warfarin interaction"}',
        '{"cite_uid":"cite-noise-a","content":"unrelated dermatology"}',
        '{"cite_uid":"cite-noise-b","content":"unrelated orthopedics"}',
    ]

    ranked = rank_documents("warfarin interaction", "", documents)

    assert len(ranked) == 2
    assert ranked[0].cite_uid == "cite-match"


def test_adaptive_cutoff_caps_candidates_at_five() -> None:
    documents = [
        f'{{"cite_uid":"cite-{index}","content":"warfarin interaction"}}'
        for index in range(7)
    ]

    ranked = rank_documents("warfarin interaction", "", documents)

    assert len(ranked) == 5
