import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

try:  # Allow importing when package is named app
    from config import settings  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from app.config import settings  # type: ignore

# Use relative import so the package works as ``adampy`` or ``app.adampy``
from ..services.ollama_client import OllamaClient


@dataclass
class Passage:
    doc_id: str
    chunk_id: str
    text: str
    title: str = ""
    url: str = ""
    section_heading: str = ""
    page_num: Optional[int] = None
    score_dense: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    collection: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_query_variants(
    ollama_client: OllamaClient,
    query: str,
    n_variants: int,
    use_hyde: bool,
) -> Dict[str, Any]:
    prompt = (
        "Rewrite the user's question for clarity while keeping meaning identical."
        " Also produce alternate phrasings using common synonyms."
        " Output JSON with keys: rewritten, alternates."
        f"\n\nQuestion: {query}"
    )
    raw = ollama_client.generate(prompt)
    rewritten = None
    alternates: List[str] = []
    try:
        data = json.loads(raw)
        rewritten = data.get("rewritten")
        alternates = data.get("alternates", [])[:n_variants]
    except Exception:
        pass

    hyde = None
    if use_hyde:
        hyde_prompt = (
            "Draft a 2-3 sentence plausible answer to retrieve semantically similar passages."
            " Keep it conservative and generic."
            f"\n\nQuestion: {query}"
        )
        hyde = ollama_client.generate(hyde_prompt)

    return {"rewritten": rewritten, "alternates": alternates, "hyde": hyde}


def dense_retrieve(
    chroma_collection, query: str, k: int, collection_name: str
) -> List[Passage]:
    res = chroma_collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    passages: List[Passage] = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for pid, doc, meta, dist in zip(ids, docs, metas, dists):
        doc_id, _, chunk_id = pid.partition(":")
        passages.append(
            Passage(
                doc_id=meta.get("doc_id", doc_id),
                chunk_id=meta.get("chunk", chunk_id),
                text=doc,
                title=meta.get("title", ""),
                url=meta.get("url") or meta.get("path", ""),
                section_heading=meta.get("section_heading") or meta.get("heading", ""),
                page_num=meta.get("page") or meta.get("page_num"),
                score_dense=1 - float(dist) if dist is not None else None,
                collection=collection_name,
            )
        )
    return passages


def rrf_fuse(lists: List[List[Passage]], k_keep: int = 50) -> List[Passage]:
    k = 60.0
    fused: Dict[tuple, Passage] = {}
    for lst in lists:
        for rank, p in enumerate(lst, start=1):
            key = (p.doc_id, p.chunk_id)
            score = 1.0 / (k + rank)
            if key not in fused:
                fused[key] = p
                fused[key].rrf_score = 0.0
            fused[key].rrf_score += score
    ranked = sorted(fused.values(), key=lambda x: x.rrf_score or 0.0, reverse=True)
    return ranked[:k_keep]



_reranker = None


def load_reranker_or_reuse():
    global _reranker
    if _reranker is None:
        from ..rerank.local_reranker import LocalCrossEncoderReranker

        _reranker = LocalCrossEncoderReranker()
    return _reranker


def rerank(
    reranker,
    query: str,
    passages: List[Passage],
    keep: int = 10,
) -> List[Passage]:
    texts = [p.text for p in passages]
    scores = reranker.score(query, texts)
    for p, s in zip(passages, scores):
        p.rerank_score = s
    ranked = sorted(passages, key=lambda x: x.rerank_score or 0.0, reverse=True)
    return ranked[:keep]


def build_grounded_answer(
    ollama_client: OllamaClient,
    query: str,
    passages: List[Passage],
) -> tuple[str, List[Dict[str, Any]]]:
    context_lines = []
    final_context: List[Dict[str, Any]] = []
    for idx, p in enumerate(passages, start=1):
        context_lines.append(f"[{idx}] {p.text}")
        final_context.append(
            {
                "citation_id": idx,
                "doc_id": p.doc_id,
                "chunk_id": p.chunk_id,
                "title": p.title,
                "url": p.url,
                "span_start": 0,
                "span_end": len(p.text),
                "collection": p.collection,
            }
        )
    context = "\n\n".join(context_lines)
    prompt = (
        "Answer only using the provided context. Cite passages with bracketed numbers [1], [2], ..."
        " matching the context items. If the answer is not supported, say you don't have enough information."
        f"\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    answer = ollama_client.generate(prompt)
    return answer, final_context
