from pathlib import Path as _Path
from typing import Any, Dict, List, Optional, Tuple

from config import (
    EMBEDDER,
    collection,
)
from doc_utils import chunk_text, read_text, sha1, sanitize_metadata


def embed(texts: List[str]) -> List[List[float]]:
    return EMBEDDER.encode(texts, normalize_embeddings=True)


def upsert_document(path: _Path, source: str) -> int:
    text = read_text(path)
    if not text:
        return 0
    try:
        collection.delete(where={"path": str(path)})
    except Exception:
        pass

    chunks = chunk_text(text)
    if not chunks:
        return 0
    embs = embed(chunks)

    doc_sha = sha1(path)
    ids = [f"{doc_sha}:{i}" for i in range(len(chunks))]
    metas = [{"source": source, "path": str(path), "chunk": i} for i in range(len(chunks))]
    collection.upsert(ids=ids, documents=chunks, metadatas=metas, embeddings=embs)
    return len(chunks)


def upsert_text(doc_id: str, text: str, base_meta: Dict[str, Any]) -> int:
    text = (text or "").strip()
    if not text:
        return 0
    try:
        collection.delete(where={"doc_id": doc_id})
    except Exception:
        pass
    chunks = chunk_text(text)
    if not chunks:
        return 0
    embs = embed(chunks)
    ids = [f"{doc_id}:{i}" for i in range(len(chunks))]
    metas = []
    for i in range(len(chunks)):
        m = sanitize_metadata(dict(base_meta))
        m["doc_id"] = doc_id
        m["chunk"] = i
        metas.append(m)
    collection.upsert(ids=ids, documents=chunks, metadatas=metas, embeddings=embs)
    return len(chunks)


def search(query: str, k: int = 4) -> List[Dict[str, Any]]:
    qemb = embed([query])[0]
    res = collection.query(query_embeddings=[qemb], n_results=k,
                           include=["documents", "metadatas", "distances"])
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    return [{"text": d, "meta": m, "score": float(1.0 / (1e-5 + dist))}
            for d, m, dist in zip(docs, metas, dists)]


def search_filtered(
    query: str,
    k: int = 4,
    path: Optional[str] = None,
    paths: Optional[List[str]] = None,
    prefix: Optional[str] = None,
    source: Optional[str] = None,
    org: Optional[str] = None,
    category: Optional[str] = None,
    doc_code: Optional[str] = None,
    owner: Optional[str] = None,
) -> List[Dict[str, Any]]:
    qemb = embed([query])[0]
    include = ["documents", "metadatas", "distances"]

    def to_hits(res):
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        return [{"text": d, "meta": m, "score": float(1.0 / (1e-5 + dist))}
                for d, m, dist in zip(docs, metas, dists)]

    results: List[Dict[str, Any]] = []

    if path:
        res = collection.query(query_embeddings=[qemb], n_results=max(k, 8),
                               include=include, where={"path": path})
        results.extend(to_hits(res))
    elif paths:
        for p in paths:
            res = collection.query(query_embeddings=[qemb], n_results=max(2, k // max(1, len(paths))) + 6,
                                   include=include, where={"path": p})
            results.extend(to_hits(res))
    elif prefix:
        try:
            res = collection.query(query_embeddings=[qemb], n_results=max(k, 12),
                                   include=include, where={"path": {"$contains": prefix}})
            results.extend(to_hits(res))
        except Exception:
            res = collection.query(query_embeddings=[qemb], n_results=max(k * 5, 20), include=include)
            results.extend([h for h in to_hits(res) if str(h["meta"].get("path", "")).startswith(prefix)])
    elif source or org or category or doc_code or owner:
        where: Dict[str, Any] = {}
        if source:   where["source"] = source
        if org:      where["org"] = org
        if category: where["category"] = category
        if doc_code: where["doc_code"] = doc_code
        if owner:    where["owner"] = owner
        res = collection.query(query_embeddings=[qemb], n_results=max(k, 12),
                               include=include, where=where)
        results.extend(to_hits(res))
    else:
        res = collection.query(query_embeddings=[qemb], n_results=k, include=include)
        results.extend(to_hits(res))

    seen: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
    for h in results:
        key = (h["meta"].get("doc_id") or h["meta"].get("path"), h["meta"].get("chunk"))
        if key not in seen or h["score"] > seen[key]["score"]:
            seen[key] = h
    hits = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:k]
    return hits
