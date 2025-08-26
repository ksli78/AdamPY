from adampy.pipeline.semantic_rag import (
    generate_query_variants,
    rrf_fuse,
    rerank,
    Passage,
)
from adampy.pipeline.citations import validate_and_fix_citations


class DummyOllama:
    def generate(self, prompt, **kwargs):
        if "Rewrite" in prompt:
            return '{"rewritten": "clear", "alternates": ["alt1", "alt2"]}'
        return "stub"


class DummyReranker:
    def score(self, q, texts):
        return [0.1 * (i + 1) for i in range(len(texts))]


def test_generate_query_variants():
    v = generate_query_variants(DummyOllama(), "question", 2, True)
    assert v["rewritten"] == "clear"
    assert len(v["alternates"]) == 2
    assert v["hyde"] == "stub"


def test_rrf_fuse():
    p1 = Passage(doc_id="d1", chunk_id="c1", text="t1")
    p2 = Passage(doc_id="d2", chunk_id="c2", text="t2")
    p3 = Passage(doc_id="d1", chunk_id="c1", text="t1")
    fused = rrf_fuse([[p1, p2], [p3]], k_keep=10)
    assert fused[0].doc_id == "d1"


def test_rerank():
    passages = [
        Passage(doc_id="d1", chunk_id="c1", text="t1"),
        Passage(doc_id="d2", chunk_id="c2", text="t2"),
    ]
    reranked = rerank(DummyReranker(), "q", passages, keep=1)
    assert len(reranked) == 1
    assert reranked[0].doc_id == "d2"


def test_validate_and_fix_citations():
    passages = [
        Passage(doc_id="d1", chunk_id="c1", text="t1"),
        Passage(doc_id="d2", chunk_id="c2", text="t2"),
    ]
    txt = "This is [1] and [99]"
    fixed, phantom, details = validate_and_fix_citations(txt, passages)
    assert phantom is True
    assert 99 in details
    assert "[1]" in fixed and "99" not in fixed


def test_query_response_fields():
    import re, pathlib

    text = pathlib.Path("main.py").read_text()
    m = re.search(r"class\s+QueryResponse\(BaseModel\):(.*?)@app.post", text, re.S)
    assert m, "QueryResponse definition missing"
    block = m.group(1)
    for key in [
        "retrieval_runs",
        "fusion",
        "rerank",
        "final_context",
        "phantom_citations_found",
    ]:
        assert key in block
