from l2_baseline.ranking import rank_documents, split_citable_documents


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


def test_nested_mcp_result_is_split_and_ranked_by_cite_uid() -> None:
    document = """{
      "source_type": "guideline",
      "title": "Hypertension Guideline",
      "items": [
        {"cite_uid": "cite-target", "content": "CKD systolic blood pressure target"},
        {"cite_uid": "cite-monitor", "content": "monitor potassium and creatinine"},
        {"cite_uid": "cite-lifestyle", "content": "diet and exercise advice"}
      ]
    }"""

    split = split_citable_documents([document])
    ranked = rank_documents("CKD blood pressure target monitoring", [document], top_k=3)

    assert {cite_uid for cite_uid, _ in split} == {
        "cite-target",
        "cite-monitor",
        "cite-lifestyle",
    }
    assert len(ranked) == 3
    assert ranked[0].cite_uid == "cite-target"
    assert all("Hypertension Guideline" in item.content for item in ranked)


def test_unstructured_result_uses_cite_uid_boundaries() -> None:
    document = (
        "source: DailyMed\n"
        "cite_uid: cite-warfarin\nwarfarin interaction section\n"
        "cite_uid: cite-storage\nstorage conditions"
    )

    split = split_citable_documents([document])

    assert [cite_uid for cite_uid, _ in split] == ["cite-warfarin", "cite-storage"]
    assert "warfarin interaction" in split[0][1]
    assert "storage conditions" not in split[0][1]


def test_duplicate_cite_uid_keeps_single_most_complete_item() -> None:
    documents = [
        '{"cite_uid":"cite-a","content":"short"}',
        '{"cite_uid":"cite-a","content":"longer clinically relevant evidence"}',
    ]

    split = split_citable_documents(documents)

    assert len(split) == 1
    assert "longer clinically relevant evidence" in split[0][1]
