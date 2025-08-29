import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

try:  # Allow importing when package is named app
    from config import settings  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from app.config import settings  # type: ignore

# Use relative import so the package works as ``adampy`` or ``app.adampy``
from ..services.ollama_client import OllamaClient


# Use the app-wide logger configured in main.py to write to journald
logger = logging.getLogger('rag')


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


def dense_retrieve(chroma_collection, query: str, k: int) -> List[Passage]:
    res = chroma_collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    logger.debug("dense_retrieve")
    passages: List[Passage] = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for pid, doc, meta, dist in zip(ids, docs, metas, dists):
        doc_id, _, chunk_id = pid.partition(":")
        url = meta.get("sp_web_url") or meta.get("url") or meta.get("path", "")
        passages.append(
            Passage(
                doc_id=meta.get("doc_id", doc_id),
                chunk_id=meta.get("chunk", chunk_id),
                text=doc,
                title=meta.get("title", ""),
                url=url,
                section_heading=meta.get("section_heading") or meta.get("heading", ""),
                page_num=meta.get("page") or meta.get("page_num"),
                score_dense=1 - float(dist) if dist is not None else None,
            )
        )
    # Log passages (placed near original line ~90)
    try:
        payload = [p.to_dict() for p in passages]
        logger.debug(
            "dense_retrieve: query=%r k=%d passages=%s",
            query,
            k,
            json.dumps(payload, ensure_ascii=False),
        )
    except Exception as e:  # pragma: no cover
        logger.debug("dense_retrieve: failed to log passages: %s", e)

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
    # After ranked = sorted(...):
    try:
        logger.debug(
            "rrf_fuse: lists=%d fused_hits=%d keep=%d",
            len(lists),
            len(ranked),
            k_keep,
        )
        for i, p in enumerate(ranked[:10]):  # first 10
            md = {} if p is None else (p.__dict__ if hasattr(p, "__dict__") else {})
            logger.debug(
                "rrf[%02d] doc_id=%s chunk_id=%s rrf_score=%.4f title=%s",
                i,
                getattr(p, "doc_id", None),
                getattr(p, "chunk_id", None),
                getattr(p, "rrf_score", float("nan")),
                (getattr(p, "title", None) or (md.get("title") if isinstance(md, dict) else None)),
            )
    except Exception as e:
        logger.debug("rrf_fuse: failed to log fused hits: %s", e)

    return ranked[:k_keep]

_reranker = None

def load_reranker_or_reuse():
    global _reranker
    if _reranker is None:
        from ..rerank.local_reranker import LocalCrossEncoderReranker

        _reranker = LocalCrossEncoderReranker()
    return _reranker


def rerank(reranker,query: str,passages: List[Passage],keep: int = 10,) -> List[Passage]:
    texts = [p.text for p in passages]
    scores = reranker.score(query, texts)

    for p, s in zip(passages, scores):
        p.rerank_score = s

    ranked = sorted(passages, key=lambda x: x.rerank_score or 0.0, reverse=True)

    try:
        logger.debug("rerank: in=%d out=%d", len(passages), min(len(passages), keep))
        for i, p in enumerate(ranked[:10]):
            logger.debug(
                "rerank[%02d] doc_id=%s chunk_id=%s score=%.4f title=%s",
                i,
                getattr(p, "doc_id", None),
                getattr(p, "chunk_id", None),
                getattr(p, "rerank_score", float("nan")),
                (p.title if hasattr(p, "title") else (p.meta or {}).get("title"))
            )
    except Exception as e:
        logger.debug("rerank: failed logging: %s", e)

    return ranked[:keep]



def build_grounded_answer(
    ollama_client: OllamaClient,
    query: str,
    passages: List[Passage],
) -> tuple[str, List[Dict[str, Any]]]:
    context_lines = []
    logger.debug("in build_grounded_answer")
    
    import re
    
    def _strip_header(t: str) -> str:
        # remove the leading DOC/PATH/FILE banner your pipeline prepends
        return re.sub(r'^DOC:.*?\nPATH:.*?\nFILE:.*?\n\n', '', t, flags=re.S)
    
    def _terms(s: str) -> set[str]:
        return {w for w in re.findall(r'\w+', (s or '').lower()) if len(w) > 1}
    
    # pick passages with any query-term overlap; if none match, keep originals
    q_terms = _terms(query)
    scored = []
    # send a smaller, more focused context (tweak N as you like)
    TOP_N = 6
    for p in passages:
        cleaned = _strip_header(p.text or "")
        p.text = cleaned  # mutate in place so everything downstream uses the clean text
        overlap = len(q_terms & _terms(cleaned))
        scored.append((overlap, p))

    # prefer passages with any query-term overlap; if none, keep originals
    scored.sort(key=lambda x: (x[0] > 0, x[0]), reverse=True)
    passages = [p for _, p in scored][:TOP_N]
    # --- NEW: collapse to the single best-matching document group ---
    from collections import defaultdict
    import math
    # --- Normalize 'scored' to a uniform [(overlap:int, Passage), ...] ---
    norm_scored = []
    for entry in scored:
        # cases: (overlap, Passage)  OR  Passage  OR anything else (ignore)
        if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], Passage):
            overlap, p = entry
            try:
                overlap = int(overlap)
            except Exception:
                overlap = 0
            norm_scored.append((overlap, p))
        elif isinstance(entry, Passage):
            norm_scored.append((0, entry))
        else:
            # unknown shape – skip
            continue

    scored = norm_scored
    if not scored:
        # nothing usable; short-circuit to original passages
        scored = [(0, p) for p in passages]

    # score by (sum of overlaps, then best rerank/dense as tie-breakers)
    by_doc = defaultdict(list)
    for overlap, p in scored:
        by_doc[p.title or p.doc_id].append((overlap, p))

    def group_score(items):
        # Normalize: ensure we are working with a list of (overlap:int, Passage) pairs
        norm = []
        for it in items:
            if isinstance(it, tuple) and len(it) >= 2 and isinstance(it[1], Passage):
                overlap, p = it[0], it[1]
            elif isinstance(it, Passage):
                overlap, p = 0, it
            else:
                # Unknown shape; skip
                continue
            try:
                overlap = int(overlap)
            except Exception:
                overlap = 0
            norm.append((overlap, p))

        # If nothing valid, score as zeroes
        if not norm:
            return (0, float("-inf"), float("-inf"))

        # Now compute scores safely
        total_overlap = sum(o for o, _ in norm)
        best_rerank  = max((getattr(p, "rerank_score", None) or float("-inf")) for _, p in norm)
        best_dense   = max((getattr(p, "score_dense",  None) or float("-inf")) for _, p in norm)
        return (total_overlap, best_rerank, best_dense)

    # pick the single best group
    best_title, best_items = max(by_doc.items(), key=group_score)

    # Sort within that doc: overlap first, then rerank, then dense
    best_items.sort(
        key=lambda t: (
            (t[0] > 0),                      # any overlap
            t[0],                            # amount of overlap
            getattr(t[1], "rerank_score", 0) or 0.0,
            getattr(t[1], "score_dense", 0) or 0.0,
        ),
        reverse=True,
    )
    
    # reduce context to tight top-k from the chosen doc
    TOP_N_FROM_BEST_DOC = 4
    passages = [p for (_, p) in best_items[:TOP_N_FROM_BEST_DOC]]

    final_context: List[Dict[str, Any]] = []
    for idx, p in enumerate(passages, start=1):
        title_part = p.title if p.title else ""
        context_lines.append(f"[{idx}] {title_part}\n{p.text}")
        final_context.append(
        {
            "citation_id": idx,
            "doc_id": p.doc_id,
            "chunk_id": p.chunk_id,
            "title": p.title,
            "url": p.url,
            "span_start": 0,
            "span_end": len(p.text),
        }
        )
    context = "\n\n".join(context_lines)
    # Log the full context for grounding (requested at line ~156)
    logger.debug("build_grounded_answer: titles=%s", "; ".join(f"[{i+1}] {p.title}" for i,p in enumerate(passages)))
    logger.debug("build_grounded_answer: context=%s", context)

    STYLE_INSTRUCTIONS = (
        "Please answer the question using only the provided context. "
        "Format the entire response as clean HTML. Use paragraphs, lists, headings, and tables when helpful. "
        "When citing, write it like: 'According to section <b>{section title}</b> [n]' instead of just '[n]'. "
        "Be detailed and natural—avoid formal or robotic phrases. "
        "If the information is not in the context, reply in a friendly way, e.g.: "
        "'I couldn’t find that information in the available documents. If you believe this should be available, "
        "please contact the IT Department for assistance.'"
    )
    prompt = (
        f"{STYLE_INSTRUCTIONS}\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    answer = ollama_client.generate(prompt)
    return answer, final_context
