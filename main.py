import sys
import pysqlite3

# Ensure Chroma uses pysqlite3 in environments where system sqlite is old
sys.modules["sqlite3"] = pysqlite3
sys.modules["sqlite3.dbapi2"] = pysqlite3.dbapi2

import os
import re
import uuid
import time
import queue
import io
import mimetypes
import threading
import subprocess
import zipfile
import base64
import tempfile
from pathlib import Path as _Path
from typing import Any, Dict, Optional, List

import requests
from fastapi import FastAPI, UploadFile, File, Query, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import traceback

import config
from config import (
    OLLAMA_URL,
    OLLAMA_KEEP_ALIVE,
    WATCH_DIR,
    UPLOAD_DIR,
    COLLECTION,
    Q_MAX,
    collection,
    SUPPORTED,
)
from models import QueryBody, ZipRequest, IngestDocument
from db import upsert_document, upsert_text, search, search_filtered
from qa import ask_with_context
from doc_utils import read_text


app = FastAPI(title="Local RAG Service")


@app.exception_handler(Exception)
async def all_exception_handler(request, exc):
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    print("\n--- Unhandled exception ---\n", tb, flush=True)
    return JSONResponse(status_code=500, content={"error": "internal_server_error", "detail": str(exc)})


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- API Endpoints ----------------
@app.post("/query")
def query_api(body: QueryBody):
    if body.org or body.category or body.doc_code or body.owner:
        hits = search_filtered(body.query, k=body.k,
                               source=None, org=body.org, category=body.category,
                               doc_code=body.doc_code, owner=body.owner)
    else:
        hits = search(body.query, k=body.k)
    answer = ask_with_context(body.query, hits, chat_history=body.history, model=body.model)
    used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if n.isdigit()})
    rich = []
    for i, h in enumerate(hits):
        meta = h.get("meta", {}) or {}
        rich.append({
            "index": i + 1,
            "title": meta.get("title"),
            "org": meta.get("org"),
            "category": meta.get("category"),
            "version": meta.get("version"),
            "revision_date": meta.get("revision_date"),
            "doc_code": meta.get("doc_code"),
            "sp_web_url": meta.get("sp_web_url"),
            "path": meta.get("path"),
            "score": h.get("score"),
            "snippet": (h.get("text") or "")[:400],
        })
    return {"answer": answer, "sources": rich}


@app.post("/upload")
def upload_api(file: UploadFile = File(...)):
    dest = _Path(UPLOAD_DIR) / file.filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        f.write(file.file.read())
    n = upsert_document(dest, source="upload")
    return {"ok": True, "chunks": n, "path": str(dest)}


@app.post("/reindex")
def reindex_api():
    root = _Path(WATCH_DIR)
    count, total_chunks = 0, 0
    if not root.exists():
        return {"files": 0, "chunks": 0}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            n = upsert_document(path, source="watched")
            total_chunks += n
            count += 1
    return {"files": count, "chunks": total_chunks}


@app.get("/status")
def status():
    try:
        cnt = collection.count()
    except Exception:
        cnt = None
    return {"ok": True, "collection": COLLECTION, "count": cnt, "supported": list(SUPPORTED)}


@app.delete("/delete")
def delete_by_path(path: str = Query(..., description="absolute file path stored in metadata")):
    try:
        collection.delete(where={"path": path})
        return {"ok": True, "path": path}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/search")
def search_only(payload: dict = Body(...)):
    q = payload.get("query", "")
    k = int(payload.get("k", 4))
    return {"results": search(q, k=k)}


@app.get("/list")
def list_docs(limit: int = 10000):
    data = collection.get(include=["metadatas"], limit=limit)
    paths = sorted({(m or {}).get("path", "unknown") for m in data.get("metadatas", [])})
    return {"count": len(paths), "paths": paths}


@app.post("/reset")
def reset_api():
    try:
        collection.delete()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/ollama_health")
def ollama_health():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return {"ok": r.status_code == 200}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/embed_health")
def embed_health():
    try:
        _ = ask_with_context("ping", [])
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/documents")
def list_documents():
    try:
        results = collection.get(include=["metadatas", "documents"], limit=10000)
        docs = []
        for i in range(len(results["ids"])):
            doc_info = {"doc_id": results["ids"][i], "metadata": results["metadatas"][i]}
            docs.append(doc_info)
        return JSONResponse(content={"documents": docs, "count": len(docs)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---- folder watcher ----
class Handler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        p = _Path(event.src_path)
        if p.suffix.lower() in SUPPORTED:
            try:
                upsert_document(p, source="watched")
            except Exception as e:
                print("ingest error:", e)

    def on_modified(self, event):
        self.on_created(event)


if _Path(WATCH_DIR).exists():
    obs = Observer()
    obs.schedule(Handler(), WATCH_DIR, recursive=True)
    obs.start()


# ---------------- Ingest queue & jobs ----------------
INGEST_Q: "queue.Queue[tuple[str,str,str]]" = queue.Queue(maxsize=Q_MAX)
JOBS: Dict[str, Dict[str, Any]] = {}


def _ingest_worker():
    while True:
        job_id, path, source = INGEST_Q.get()
        try:
            t0 = time.time()
            chunks = upsert_document(_Path(path), source=source)
            JOBS[job_id] = {"status": "done", "path": path, "chunks": chunks, "ms": int((time.time() - t0) * 1000)}
        except Exception as e:
            JOBS[job_id] = {"status": "error", "path": path, "error": str(e)}
        finally:
            INGEST_Q.task_done()


def _start_worker():
    threading.Thread(target=_ingest_worker, daemon=True).start()


_start_worker()


@app.post("/upload_async")
def upload_async_api(file: UploadFile = File(...)):
    dest = _Path(UPLOAD_DIR) / file.filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        f.write(file.file.read())
    jid = str(uuid.uuid4())
    JOBS[jid] = {"status": "queued", "path": str(dest)}
    try:
        INGEST_Q.put_nowait((jid, str(dest), "upload"))
    except queue.Full:
        JOBS[jid] = {"status": "error", "path": str(dest), "error": "ingest queue is full"}
    return {"ok": True, "job_id": jid, "path": str(dest)}


@app.get("/jobs")
def list_jobs():
    keys = list(JOBS.keys())[-200:]
    return {k: JOBS[k] for k in keys}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    return JOBS.get(job_id, {"status": "unknown"})


# ---------------- Downloads ----------------
def _is_under_allowed(p: _Path) -> bool:
    roots = [_Path(UPLOAD_DIR).resolve(), _Path(WATCH_DIR).resolve()]
    try:
        rp = p.resolve()
        for r in roots:
            try:
                if rp.is_relative_to(r.resolve()):
                    return True
            except AttributeError:
                if str(rp).startswith(str(r.resolve()) + os.sep):
                    return True
        return False
    except Exception:
        return False


@app.get("/download")
def download(path: str, inline: bool = False):
    p = _Path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "file not found")
    if not _is_under_allowed(p):
        raise HTTPException(403, "forbidden path")

    mt, _ = mimetypes.guess_type(str(p))
    resp = FileResponse(path=str(p), media_type=mt or "application/octet-stream", filename=p.name)
    if inline:
        resp.headers["Content-Disposition"] = f'inline; filename="{p.name}"'
    return resp


@app.post("/download_zip")
def download_zip(req: ZipRequest):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for raw in req.paths or []:
            p = _Path(raw)
            if not (p.exists() and p.is_file() and _is_under_allowed(p)):
                continue
            z.write(str(p), arcname=p.name)
    buf.seek(0)
    headers = {"Content-Disposition": 'attachment; filename="documents.zip"'}
    return StreamingResponse(buf, media_type="application/zip", headers=headers)


@app.post("/query_path")
def query_path(body: Dict[str, Any] = Body(...)):
    query = body.get("query", "")
    k = int(body.get("k", 4))
    history = body.get("history")
    path = body.get("path")
    paths = body.get("paths")
    prefix = body.get("prefix")
    source = body.get("source")
    model = body.get("model")
    org = body.get("org")
    category = body.get("category")
    doc_code = body.get("doc_code")
    owner = body.get("owner")

    hits = search_filtered(query, k=k, path=path, paths=paths, prefix=prefix, source=source,
                           org=org, category=category, doc_code=doc_code, owner=owner)
    answer = ask_with_context(query, hits, chat_history=history, model=model)
    used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if n.isdigit()})
    filtered = [hits[i-1] for i in used if 1 <= i <= len(hits)] if used else []
    return {"answer": answer, "sources": filtered}


@app.post("/ingest_document")
def ingest_document_api(body: IngestDocument):
    orig_size, orig_overlap = config.CHUNK_SIZE, config.CHUNK_OVERLAP
    try:
        if body.chunk_size:
            config.CHUNK_SIZE = int(body.chunk_size)
        if body.chunk_overlap:
            config.CHUNK_OVERLAP = int(body.chunk_overlap)

        version_key = body.version_label or body.etag or "v1"
        spid = str(body.sp_item_id or body.sp_file_id or body.sp_web_url)
        doc_id = f"{spid}:{version_key}"

        base_meta = {
            "source": "sharepoint",
            "sp_site_id": body.sp_site_id,
            "sp_list_id": body.sp_list_id,
            "sp_item_id": body.sp_item_id,
            "sp_drive_id": body.sp_drive_id,
            "sp_file_id": body.sp_file_id,
            "sp_web_url": body.sp_web_url,
            "etag": body.etag,
            "version_label": body.version_label,
            "title": body.title,
            "doc_code": body.doc_code,
            "org_code": body.org_code,
            "org": body.org,
            "category": body.category,
            "owner": body.owner,
            "version": body.version,
            "revision_date": body.revision_date,
            "latest_review_date": body.latest_review_date,
            "document_review_date": body.document_review_date,
            "review_approval_date": body.review_approval_date,
            "keywords": body.keywords or [],
            "enterprise_keywords": body.enterprise_keywords or [],
            "association_ids": body.association_ids or [],
            "domain": body.domain or "HR",
            "allowed_groups": body.allowed_groups or ["AllEmployees"],
        }

        if body.text_content and str(body.text_content).strip():
            n = upsert_text(doc_id, body.text_content, base_meta)
            return {"ok": True, "doc_id": doc_id, "chunks": n, "used": "text_content"}

        if body.content_bytes:
            raw = base64.b64decode(body.content_bytes)
            suffix = _Path(body.file_name).suffix if body.file_name else ""
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
                tmp = _Path(tf.name)
                tf.write(raw)
            try:
                text = read_text(tmp)
            finally:
                try:
                    tmp.unlink(missing_ok=True)
                except TypeError:
                    if tmp.exists():
                        tmp.unlink()
            n = upsert_text(doc_id, text, base_meta)
            return {"ok": True, "doc_id": doc_id, "chunks": n, "used": "content_bytes"}

        raise HTTPException(status_code=400, detail="Provide either text_content or content_bytes.")
    finally:
        config.CHUNK_SIZE, config.CHUNK_OVERLAP = orig_size, orig_overlap


@app.delete("/delete_sp")
def delete_by_sp(sp_item_id: Optional[str] = Query(None),
                 sp_file_id: Optional[str] = Query(None),
                 version: Optional[str] = Query(None),
                 etag: Optional[str] = Query(None)):
    where: Dict[str, Any] = {}
    if sp_item_id:
        where["sp_item_id"] = sp_item_id
    if sp_file_id:
        where["sp_file_id"] = sp_file_id
    if not where:
        raise HTTPException(400, "Provide sp_item_id or sp_file_id")
    if version:
        where["version_label"] = version
    if etag:
        where["etag"] = etag
    try:
        collection.delete(where=where)
        return {"ok": True, "where": where}
    except Exception as e:
        return {"ok": False, "error": str(e), "where": where}
