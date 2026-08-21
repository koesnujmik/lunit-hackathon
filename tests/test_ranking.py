from l2_baseline.ranking import rank_documents


def test_tfidf_returns_only_citable_top_three() -> None:
    passage = "만성 신장질환 chronic kidney disease blood pressure target"
    documents = [
        '{"cite_uid":"cite-a","content":"chronic kidney disease blood pressure target"}',
        '{"cite_uid":"cite-b","content":"blood pressure in CKD"}',
        '{"cite_uid":"cite-c","content":"kidney guideline"}',
        '{"cite_uid":"cite-d","content":"unrelated dermatology"}',
        '{"content":"highly similar chronic kidney disease but not citable"}',
    ]
    ranked = rank_documents(passage, documents, top_k=3)
    assert len(ranked) == 3
    assert ranked[0].cite_uid == "cite-a"
    assert all(item.cite_uid != "cite-d" for item in ranked[:2])


def test_tfidf_empty_documents() -> None:
    assert rank_documents("query", [], top_k=3) == []
