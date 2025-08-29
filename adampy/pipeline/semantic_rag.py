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
    from html import escape as _htmlesc
    from collections import defaultdict

    # ---------- helpers (local) ----------
    def _strip_header(t: str) -> str:
        # remove the leading DOC/PATH/FILE banner your pipeline prepends
        return re.sub(r'^DOC:.*?\nPATH:.*?\nFILE:.*?\n\n', '', t, flags=re.S)

    def _terms(s: str) -> set[str]:
        return {w for w in re.findall(r'\w+', (s or '').lower()) if len(w) > 1}

    def _force_html(s: str) -> str:
        # If it already looks like HTML, keep it.
        if re.search(r'</?(p|ul|ol|li|h[1-6]|table|thead|tbody|tr|td|th|br)\b', s, re.I):
            return s
        # Convert plaintext/markdown-ish breaks to simple HTML
        parts = [p.strip() for p in re.split(r'\n\s*\n', s.strip()) if p.strip()]
        if not parts:
            return "<p></p>"
        # Turn simple bullets into <ul> if appropriate
        lines = [ln for ln in s.splitlines() if ln.strip()]
        if lines and all(re.match(r'^\s*[-*]\s+', ln) for ln in lines):
            items = [re.sub(r'^\s*[-*]\s+', '', ln).strip() for ln in lines]
            return "<ul>" + "".join(f"<li>{_htmlesc(it)}</li>" for it in items) + "</ul>"
        return "".join(f"<p>{_htmlesc(p)}</p>" for p in parts)

    def _upgrade_citation_sup(s: str) -> str:
        # Replace bare [number] (not inside tags) with <sup>[number]</sup>
        return re.sub(r'(?<![>#])\[(\d+)\]', r'<sup>[\1]</sup>', s)

    def _dehedge_anywhere(s: str) -> str:
        # Remove robotic hedges anywhere (case-insensitive)
        patterns = [
            r'\baccording to (the )?provided context\b',
            r'\bbased on (the )?provided context\b',
            r'\baccording to (the )?context\b',
            r'\bas per (the )?context\b',
            r'\bfrom (the )?context\b',
            r'\bthe context (does|doesn’t|does not|states|indicates)\b',
        ]
        for pat in patterns:
            s = re.sub(pat, '', s, flags=re.I)
        # Clean punctuation/spacing after removals
        s = re.sub(r'\s{2,}', ' ', s)
        s = re.sub(r'\s+([,.;:])', r'\1', s)
        return s.strip()

    def _inject_section_titles(s: str, id_to_title: dict[str, str]) -> str:
        """
        If model wrote 'According to section [n]' (without a title),
        replace with 'According to section <b>{title}</b> [n]'.
        Also, if it wrote a wrong/empty title, prefer our known title.
        """
        # Case 1: No title at all
        def repl_no_title(m):
            idx = m.group(1)
            title = id_to_title.get(idx, "").strip()
            if not title:
                return f"According to section [\u200b{idx}]"  # zero-width space to avoid re-match loop
            return f"According to section <b>{_htmlesc(title)}</b> [{idx}]"

        s = re.sub(r'(?i)according to section\s*\[(\d+)\]', repl_no_title, s)

        # Case 2: Has a title but we want to normalize it to our canonical title
        def repl_has_title(m):
            idx = m.group(2)
            want = id_to_title.get(idx, "").strip()
            if not want:
                return m.group(0)  # keep original if we don't know the title
            return f"According to section <b>{_htmlesc(want)}</b> [{idx}]"

        s = re.sub(
            r'(?i)according to section\s*(?:<b>)?([^<\[]*?)(?:</b>)?\s*\[(\d+)\]',
            repl_has_title,
            s,
        )
        return s

    # ---------- passage selection & grouping ----------
    q_terms = _terms(query)
    scored = []
    TOP_N = 6
    for p in passages:
        cleaned = _strip_header(p.text or "")
        p.text = cleaned  # mutate so downstream uses clean text
        overlap = len(q_terms & _terms(cleaned))
        scored.append((overlap, p))

    # prefer passages with any query-term overlap; if none, keep originals
    scored.sort(key=lambda x: (x[0] > 0, x[0]), reverse=True)
    passages = [p for _, p in scored][:TOP_N]

    # group by doc/title and score groups
    from collections import defaultdict
    by_doc = defaultdict(list)

    # normalize 'scored'
    norm_scored = []
    for entry in scored:
        if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], Passage):
            try:
                overlap = int(entry[0])
            except Exception:
                overlap = 0
            norm_scored.append((overlap, entry[1]))
        elif isinstance(entry, Passage):
            norm_scored.append((0, entry))
    if not norm_scored:
        norm_scored = [(0, p) for p in passages]

    for overlap, p in norm_scored:
        by_doc[p.title or p.doc_id].append((overlap, p))

    def group_score(items):
        if not items:
            return (0, float("-inf"), float("-inf"))
        total_overlap = sum(o for o, _ in items)
        best_rerank  = max((getattr(p, "rerank_score", None) or float("-inf")) for _, p in items)
        best_dense   = max((getattr(p, "score_dense",  None) or float("-inf")) for _, p in items)
        return (total_overlap, best_rerank, best_dense)

    best_title, best_items = max(by_doc.items(), key=group_score)

    # sort within that doc
    best_items.sort(
        key=lambda t: (
            (t[0] > 0),
            t[0],
            getattr(t[1], "rerank_score", 0) or 0.0,
            getattr(t[1], "score_dense", 0) or 0.0,
        ),
        reverse=True,
    )

    # reduce context to tight top-k from the chosen doc
    TOP_N_FROM_BEST_DOC = 4
    passages = [p for (_, p) in best_items[:TOP_N_FROM_BEST_DOC]]

    # ---------- build context ----------
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
    logger.debug("build_grounded_answer: titles=%s", "; ".join(f"[{i+1}] {p.title}" for i,p in enumerate(passages)))
    logger.debug("build_grounded_answer: context=%s", context)

    # ---------- prompt ----------
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

    # ---------- generate & enforce ----------
    answer = ollama_client.generate(prompt)

    # Build id->title map for citation title injection
    id_to_title = {str(c['citation_id']): (c.get('title') or '').strip() for c in final_context}

    # 1) Ensure citation phrasing includes a title where missing / normalize titles
    answer = _inject_section_titles(answer, id_to_title)
    # 2) Remove hedgy "context" phrasing anywhere
    answer = _dehedge_anywhere(answer)
    # 3) Better-looking citations
    answer = _upgrade_citation_sup(answer)
    # 4) Guarantee HTML
    answer = _force_html(answer)

    return answer, final_context


# def build_grounded_answer(
#     ollama_client: OllamaClient,
#     query: str,
#     passages: List[Passage],
# ) -> tuple[str, List[Dict[str, Any]]]:
#     context_lines = []
#     logger.debug("in build_grounded_answer")
    
#     import re
    
#     def _strip_header(t: str) -> str:
#         # remove the leading DOC/PATH/FILE banner your pipeline prepends
#         return re.sub(r'^DOC:.*?\nPATH:.*?\nFILE:.*?\n\n', '', t, flags=re.S)
    
#     def _terms(s: str) -> set[str]:
#         return {w for w in re.findall(r'\w+', (s or '').lower()) if len(w) > 1}
    
#     # pick passages with any query-term overlap; if none match, keep originals
#     q_terms = _terms(query)
#     scored = []
#     # send a smaller, more focused context (tweak N as you like)
#     TOP_N = 6
#     for p in passages:
#         cleaned = _strip_header(p.text or "")
#         p.text = cleaned  # mutate in place so everything downstream uses the clean text
#         overlap = len(q_terms & _terms(cleaned))
#         scored.append((overlap, p))

#     # prefer passages with any query-term overlap; if none, keep originals
#     scored.sort(key=lambda x: (x[0] > 0, x[0]), reverse=True)
#     passages = [p for _, p in scored][:TOP_N]
#     # --- NEW: collapse to the single best-matching document group ---
#     from collections import defaultdict
#     import math
#     # --- Normalize 'scored' to a uniform [(overlap:int, Passage), ...] ---
#     norm_scored = []
#     for entry in scored:
#         # cases: (overlap, Passage)  OR  Passage  OR anything else (ignore)
#         if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], Passage):
#             overlap, p = entry
#             try:
#                 overlap = int(overlap)
#             except Exception:
#                 overlap = 0
#             norm_scored.append((overlap, p))
#         elif isinstance(entry, Passage):
#             norm_scored.append((0, entry))
#         else:
#             # unknown shape – skip
#             continue

#     scored = norm_scored
#     if not scored:
#         # nothing usable; short-circuit to original passages
#         scored = [(0, p) for p in passages]

#     # score by (sum of overlaps, then best rerank/dense as tie-breakers)
#     by_doc = defaultdict(list)
#     for overlap, p in scored:
#         by_doc[p.title or p.doc_id].append((overlap, p))

#     def group_score(items):
#         # Normalize: ensure we are working with a list of (overlap:int, Passage) pairs
#         norm = []
#         for it in items:
#             if isinstance(it, tuple) and len(it) >= 2 and isinstance(it[1], Passage):
#                 overlap, p = it[0], it[1]
#             elif isinstance(it, Passage):
#                 overlap, p = 0, it
#             else:
#                 # Unknown shape; skip
#                 continue
#             try:
#                 overlap = int(overlap)
#             except Exception:
#                 overlap = 0
#             norm.append((overlap, p))

#         # If nothing valid, score as zeroes
#         if not norm:
#             return (0, float("-inf"), float("-inf"))

#         # Now compute scores safely
#         total_overlap = sum(o for o, _ in norm)
#         best_rerank  = max((getattr(p, "rerank_score", None) or float("-inf")) for _, p in norm)
#         best_dense   = max((getattr(p, "score_dense",  None) or float("-inf")) for _, p in norm)
#         return (total_overlap, best_rerank, best_dense)

#     # pick the single best group
#     best_title, best_items = max(by_doc.items(), key=group_score)

#     # Sort within that doc: overlap first, then rerank, then dense
#     best_items.sort(
#         key=lambda t: (
#             (t[0] > 0),                      # any overlap
#             t[0],                            # amount of overlap
#             getattr(t[1], "rerank_score", 0) or 0.0,
#             getattr(t[1], "score_dense", 0) or 0.0,
#         ),
#         reverse=True,
#     )
    
#     # reduce context to tight top-k from the chosen doc
#     TOP_N_FROM_BEST_DOC = 4
#     passages = [p for (_, p) in best_items[:TOP_N_FROM_BEST_DOC]]

#     final_context: List[Dict[str, Any]] = []
#     for idx, p in enumerate(passages, start=1):
#         title_part = p.title if p.title else ""
#         context_lines.append(f"[{idx}] {title_part}\n{p.text}")
#         final_context.append(
#         {
#             "citation_id": idx,
#             "doc_id": p.doc_id,
#             "chunk_id": p.chunk_id,
#             "title": p.title,
#             "url": p.url,
#             "span_start": 0,
#             "span_end": len(p.text),
#         }
#         )
#     context = "\n\n".join(context_lines)
#     # Log the full context for grounding (requested at line ~156)
#     logger.debug("build_grounded_answer: titles=%s", "; ".join(f"[{i+1}] {p.title}" for i,p in enumerate(passages)))
#     logger.debug("build_grounded_answer: context=%s", context)

#     prompt = (
#         "Please answer the question using only the provided context. "
#         "Format the entire response as clean HTML. Use paragraphs, lists, headings, and tables when helpful. "
#         "When citing, write it like: 'According to section <b>{section title}</b> [n]' instead of just '[n]'. "
#         "Be detailed and natural—avoid formal or robotic phrases. "
#         "If the information is not in the context, reply in a friendly way, e.g.: "
#         "'I couldn’t find that information in the available documents. If you believe this should be available, "
#         "please contact the IT Department for assistance.' "
#         f"\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
#     )

#     answer = ollama_client.generate(prompt)
  
    
#     return answer, final_context
