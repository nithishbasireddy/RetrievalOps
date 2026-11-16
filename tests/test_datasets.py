import os
import unicodedata
from pathlib import Path

import pytest

from retrievalops.config import DatasetConfig
from retrievalops.datasets import load_dataset

REPO = Path(__file__).resolve().parent.parent
LBR = Path(os.environ.get("LBR_PATH", REPO.parent / "datasets" / "legalbenchrag"))


def test_local_legal_loads_all_documents():
    ds = load_dataset(DatasetConfig(name="local_legal", path=str(REPO / "data")))
    assert len(ds.documents) == 5
    assert len(ds.queries) == 10
    assert all(d.text.strip() for d in ds.documents)
    assert all(q.reference_answer for q in ds.queries)


@pytest.mark.skipif(not LBR.exists(), reason="LegalBench-RAG not downloaded")
def test_legalbench_rag_spans_match_answers():
    import json

    ds = load_dataset(DatasetConfig(name="legalbench_rag", path=str(LBR)))
    assert len(ds.queries) == 6889
    docs = {d.doc_id: d.text for d in ds.documents}
    # every gold span lies inside its document and equals the stored answer text
    for subset in ["contractnli", "cuad", "maud", "privacy_qa"]:
        tests = json.loads((LBR / "benchmarks" / f"{subset}.json").read_text())["tests"]
        for t in tests:
            for s in t["snippets"]:
                text = docs[unicodedata.normalize("NFC", s["file_path"])]
                a, b = s["span"]
                assert 0 <= a < b <= len(text)
                assert text[a:b] == s["answer"]


@pytest.mark.skipif(not LBR.exists(), reason="LegalBench-RAG not downloaded")
def test_lbr_mini_matches_reference_protocol():
    ds = load_dataset(DatasetConfig(name="legalbench_rag", path=str(LBR), sampling="lbr_mini"))
    assert len(ds.queries) == 4 * 194
    used = {s.doc_id for q in ds.queries for s in q.gold_spans}
    assert {d.doc_id for d in ds.documents} == used
    again = load_dataset(DatasetConfig(name="legalbench_rag", path=str(LBR), sampling="lbr_mini"))
    assert [q.qid for q in again.queries] == [q.qid for q in ds.queries]


@pytest.mark.skipif(not LBR.exists(), reason="LegalBench-RAG not downloaded")
def test_query_file_replaces_queries_keeps_corpus(tmp_path):
    import json

    full = load_dataset(DatasetConfig(name="legalbench_rag", path=str(LBR)))
    doc = full.documents[0]
    subset = doc.doc_id.split("/")[0]
    qf = tmp_path / "q.json"
    qf.write_text(json.dumps({"tests": [
        {"qid": "s/0", "subset": subset, "query": "hello?", "snippets": [{"file_path": doc.doc_id, "span": [0, 10]}]}
    ]}))
    ds = load_dataset(DatasetConfig(name="legalbench_rag", path=str(LBR), query_file=str(qf)))
    assert [q.qid for q in ds.queries] == ["s/0"]
    assert [d.doc_id for d in ds.documents] == [d.doc_id for d in full.documents]
