import sys
import pysqlite3

# Ensure Chroma uses pysqlite3 (bundled SQLite) in environments where system sqlite is old
sys.modules["sqlite3"] = pysqlite3
sys.modules["sqlite3.dbapi2"] = pysqlite3.dbapi2
 
import os
import re
import io
import uuid
import time
import queue
import json
import hashlib
import mimetypes
import threading
import subprocess
import zipfile
from pathlib import Path
from typing import List, Optional, Dict, Any

import requests
from fastapi import FastAPI, UploadFile, File, Query, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# ---------------- Vector DB ----------------
import chromadb

# ---------------- File watching ----------------
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ---------------- Document parsing ----------------
from pypdf import PdfReader
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path
import pytesseract
from bs4 import BeautifulSoup
from PIL import Image
import openpyxl
import extract_msg

from fastapi.responses import JSONResponse
import traceback

# ---------------- Embeddings: nomic-embed-text via ONNXRuntime ----------------
import onnxruntime as ort
from tokenizers import Tokenizer
import numpy as np

class NomicOnnxEmbedder:
    """
    ONNXRuntime wrapper for nomic-ai/nomic-embed-text local model.

    Expected files under EMBED_MODEL_DIR:
      - tokenizer.json
      - onnx/model.onnx   (change onnx_filename if you want a different variant)
    """
    def __init__(self, model_dir: str, max_len: int = None, onnx_filename: str = "model.onnx"):
        import os
        self.model_dir = model_dir.rstrip("/")
        tok_path = os.path.join(self.model_dir, "tokenizer.json")
        onnx_path = os.path.join(self.model_dir, "onnx", onnx_filename)

        if not os.path.exists(tok_path):
            raise RuntimeError(f"Tokenizer not found: {tok_path}")
        if not os.path.exists(onnx_path):
            raise RuntimeError(f"ONNX model not found: {onnx_path}")

        self.tokenizer = Tokenizer.from_file(tok_path)
        self.max_len = max_len or int(os.getenv("EMBED_MAX_LEN", "2048"))

        # CPU by default; switch to onnxruntime-gpu + CUDA providers if desired
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)

        # Model I/O signatures
        self.input_names = [i.name for i in self.session.get_inputs()]
        outs = [o.name for o in self.session.get_outputs()]
        if not outs:
            raise RuntimeError("Could not resolve ONNX output name.")
        self.output_name = outs[0]  # often 'last_hidden_state'

    def _prepare_arrays(self, texts):
        max_len = self.max_len
        input_ids, attention_mask = [], []
        for t in texts:
            enc = self.tokenizer.encode(t or "")
            ids = enc.ids[:max_len]
            att = [1] * len(ids)
            if len(ids) < max_len:
                pad = max_len - len(ids)
                ids += [0] * pad
                att += [0] * pad
            input_ids.append(ids)
            attention_mask.append(att)

        # Optional inputs: many BERT-style models require these
        token_type_ids = [[0] * max_len for _ in texts]          # all zeros
        position_ids   = [list(range(max_len)) for _ in texts]   # sometimes expected

        arrays = {
            "input_ids":      np.asarray(input_ids, dtype=np.int64),
            "attention_mask": np.asarray(attention_mask, dtype=np.int64),
            "token_type_ids": np.asarray(token_type_ids, dtype=np.int64),
            "position_ids":   np.asarray(position_ids, dtype=np.int64),
        }
        return arrays

    def _masked_mean_pool(self, token_embs: np.ndarray, attn: np.ndarray) -> np.ndarray:
        """
        token_embs: [B, T, H], attn: [B, T]
        returns: [B, H]
        """
        attn = attn[:, :token_embs.shape[1]].astype(np.float32)
        attn_exp = np.expand_dims(attn, axis=-1)                 # [B, T, 1]
        summed = (token_embs * attn_exp).sum(axis=1)             # [B, H]
        counts = np.clip(attn_exp.sum(axis=1), 1e-6, None)       # [B, 1]
        return summed / counts

    def encode(self, texts, normalize_embeddings: bool = True):
        if not isinstance(texts, list):
            texts = [texts]

        arrays = self._prepare_arrays(texts)

        # Feed ONLY what the model actually requires
        feed = {name: arrays[name] for name in self.input_names if name in arrays}
        missing = [name for name in self.input_names if name not in feed]
        if missing:
            raise RuntimeError(f"Missing inputs not covered by adapter: {missing}")

        out = self.session.run([self.output_name], feed)[0]
        vecs = np.asarray(out, dtype=np.float32)

        # Pool if the model returned token embeddings
        if vecs.ndim == 3:                     # [B, T, H]
            vecs = self._masked_mean_pool(vecs, arrays["attention_mask"])
        elif vecs.ndim == 4:
            vecs = vecs.squeeze()
            if vecs.ndim == 3:
                vecs = self._masked_mean_pool(vecs, arrays["attention_mask"])
        elif vecs.ndim == 1:                   # [H] -> [1, H]
            vecs = vecs[None, :]

        # L2 normalize
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
            vecs = vecs / norms

        return vecs.tolist()



# ---------------- Configuration ----------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
EMBED_MODEL_DIR = os.getenv("EMBED_MODEL_DIR", "/opt/adam/models/nomic-ai/nomic-embed-text")
CHAT_MODEL = os.getenv("CHAT_MODEL", "adam-large:latest")  # default to Adam Large
CHROMA_DIR = os.getenv("CHROMA_DIR", "/srv/rag/chroma")
WATCH_DIR = os.getenv("WATCH_DIR", "/srv/rag/watched")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/srv/rag/uploads")
COLLECTION = os.getenv("COLLECTION", "company_docs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
Q_MAX = int(os.getenv("INGEST_QUEUE_MAX", "8"))

# ---------------- Model aliasing ----------------
ALIAS_MAP = {
    "Adam Large": "adam-large:latest",
    "Adam Lite": "adam-lite:latest",
    "adam-large": "adam-large:latest",
    "adam-lite": "adam-lite:latest",
    "llama3.3:70b": "llama3.3:70b",
    "llama3:8b": "llama3:8b",
}
DEFAULT_MODEL = CHAT_MODEL


def resolve_model(name: Optional[str]) -> str:
    """Return an Ollama model tag given a friendly name or raw tag."""
    if not name:
        return DEFAULT_MODEL
    return ALIAS_MAP.get(name, name)


# Initialize embedder early so startup fails fast if model missing
EMBEDDER = NomicOnnxEmbedder(EMBED_MODEL_DIR)

# ---------------- FastAPI app ----------------
app = FastAPI(title="Local RAG Service")

@app.exception_handler(Exception)
async def all_exception_handler(request, exc):
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    # Log the full traceback to stdout/journalctl
    print("\n--- Unhandled exception ---\n", tb, flush=True)
    # Return structured JSON so clients don't see plain text "Internal Server Error"
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": str(exc)},
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(COLLECTION)

# ---------------- Helpers ----------------
SUPPORTED = (
    ".pdf", ".docx", ".txt",
    ".md", ".csv", ".log", ".json", ".py", ".cs", ".java",
    ".html", ".htm",
    ".eml", ".msg",
    ".xlsx", ".pptx",
    ".png", ".jpg", ".jpeg", ".tiff", ".tif",
    ".doc", ".rtf",
)


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path: Path) -> str:
    ext = path.suffix.lower()

    if ext == ".pdf":
        # pypdf
        try:
            reader = PdfReader(str(path))
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
            if text and len(text.strip()) > 50:
                return text
        except Exception:
            pass
        # pdftotext
        try:
            out = subprocess.run(["pdftotext", "-layout", str(path), "-"],
                                 capture_output=True, text=True, timeout=60)
            if out.stdout and len(out.stdout.strip()) > 50:
                return out.stdout
        except Exception:
            pass
        # OCR
        try:
            images = convert_from_path(str(path))
            ocr_text = []
            for img in images:
                ocr_text.append(pytesseract.image_to_string(img))
            joined = "\n".join(ocr_text)
            if joined.strip():
                return joined
        except Exception:
            pass
        return ""

    elif ext == ".docx":
        try:
            doc = Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return ""

    elif ext in (".txt", ".md", ".csv", ".log", ".json", ".py", ".cs", ".java"):
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""

    elif ext in (".html", ".htm"):
        try:
            html = path.read_text(encoding="utf-8", errors="ignore")
            soup = BeautifulSoup(html, "lxml")
            return soup.get_text(separator="\n")
        except Exception:
            return ""

    elif ext == ".eml":
        try:
            import email
            msg = email.message_from_bytes(path.read_bytes())
            parts = []
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    parts.append(part.get_payload(decode=True).decode(errors="ignore"))
            return "\n".join(parts)
        except Exception:
            return ""

    elif ext == ".msg":
        try:
            m = extract_msg.Message(str(path))
            text = [m.subject or "", m.body or ""]
            return "\n".join([t for t in text if t])
        except Exception:
            return ""

    elif ext == ".xlsx":
        try:
            wb = openpyxl.load_workbook(str(path), data_only=True)
            parts = []
            for ws in wb.worksheets:
                for row in ws.iter_rows(values_only=True):
                    parts.append(" ".join([str(c) for c in row if c is not None]))
            return "\n".join(parts)
        except Exception:
            return ""

    elif ext == ".pptx":
        try:
            from pptx import Presentation
            prs = Presentation(str(path))
            parts = []
            for slide in prs.slides:
                for shp in slide.shapes:
                    if hasattr(shp, "text") and shp.text:
                        parts.append(shp.text)
            return "\n".join(parts)
        except Exception:
            return ""

    elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif"):
        try:
            img = Image.open(str(path))
            return pytesseract.image_to_string(img)
        except Exception:
            return ""

    elif ext == ".doc":
        # try antiword
        try:
            out = subprocess.run(["antiword", str(path)], capture_output=True, text=True, timeout=60)
            if out.stdout.strip():
                return out.stdout
        except Exception:
            pass
        # fallback: convert to pdf via libreoffice headless
        try:
            subprocess.run(["libreoffice", "--headless", "--convert-to", "pdf",
                            "--outdir", str(path.parent), str(path)], check=True, timeout=120)
            pdf_path = path.with_suffix(".pdf")
            if pdf_path.exists():
                return read_text(pdf_path)
        except Exception:
            pass
        return ""

    elif ext == ".rtf":
        try:
            out = subprocess.run(["unrtf", "--text", str(path)], capture_output=True, text=True, timeout=60)
            return out.stdout
        except Exception:
            return ""

    else:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text.strip()
    if not text:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def embed(texts: List[str]) -> List[List[float]]:
    return EMBEDDER.encode(texts, normalize_embeddings=True)


def upsert_document(path: Path, source: str) -> int:
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


def search(query: str, k: int = 4) -> List[Dict[str, Any]]:
    qemb = embed([query])[0]
    res = collection.query(query_embeddings=[qemb], n_results=k,
                           include=["documents", "metadatas", "distances"])
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    return [{"text": d, "meta": m, "score": float(1.0 / (1e-5 + dist))}
            for d, m, dist in zip(docs, metas, dists)]


def _dehedge(text: str) -> str:
    patterns = [
        r'(?i)^\s*(according to|based on|from)\s+(the\s+)?(provided|given)\s+(context|information|documents)\s*[:,\-]*\s*',
        r'(?i)^\s*(according to|based on)\s+(the\s+)?document(s)?\s*[:,\-]*\s*',
    ]
    for p in patterns:
        text = re.sub(p, "", text, count=1)
    return text.strip()

def ask_with_context(question: str, hits: List[dict], chat_history: Optional[List[dict]] = None, model: Optional[str] = None) -> str:
    ql = (question or "").lower()

    meta_triggers = [
        "your name", "what is your name", "what's your name",
        "who are you", "who is adam", "what does adam stand for",
        "what are you", "introduce yourself"
    ]
    if any(t in ql for t in meta_triggers):
        return "I am Adam - the Amentum Document and Assistance Model (ADAM)."

    context = "\n\n".join([f"[{i+1}] {h['text']}" for i, h in enumerate(hits)])

    sys_prompt = (
        "You are Adam — the Amentum Document and Assistance Model (ADAM). "
        "Answer directly and succinctly. Do not start with phrases like "
        "'According to the provided context'. Use ONLY the provided context for factual claims and insert "
        "inline bracket citations like [1], [2] right after the sentence they support. "
        "Do not append a 'Sources:' section. If the answer is not in the context, say you do not know."
    )

    messages = [{"role": "system", "content": sys_prompt}]
    if chat_history:
        messages.extend(chat_history)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    messages.append({"role": "user", "content": prompt})

    model_tag = resolve_model(model)

    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": model_tag, "messages": messages, "stream": False,
                  "options": {"temperature": 0.2, "num_predict": 300}},
            timeout=120
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama connection error: {e}")

    if r.status_code != 200:
        body = r.text
        if len(body) > 800:
            body = body[:800] + "...(truncated)"
        raise HTTPException(status_code=502, detail=f"Ollama status {r.status_code}: {body}")

    try:
        j = r.json()
    except Exception:
        body = r.text
        if len(body) > 800:
            body = body[:800] + "...(truncated)"
        raise HTTPException(status_code=502, detail=f"Ollama returned non-JSON: {body}")

    msg = j.get("message") or {}
    content = msg.get("content") or j.get("content") or ""

    if not content.strip() and hits:
        top = (hits[0].get("text") or "").strip()
        if top and len(top) <= 200 and any(p in ql for p in ["what does", "say", "content", "quote", "exact text"]):
            return f'It says: "{{ " ".join(top.split()) }}" [1].'

    return _dehedge(content)

# ---- API models & endpoints ----
class QueryBody(BaseModel):
    query: str
    k: int = 4
    history: Optional[List[dict]] = None
    model: Optional[str] = None  # "Adam Large", "Adam Lite", or raw Ollama tag


@app.post("/query")
def query_api(body: QueryBody):
    hits = search(body.query, k=body.k)
    answer = ask_with_context(body.query, hits, chat_history=body.history, model=body.model)
    used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if n.isdigit()})
    filtered = [hits[i-1] for i in used if 1 <= i <= len(hits)] if used else []
    return {"answer": answer, "sources": filtered}


@app.post("/upload")
def upload_api(file: UploadFile = File(...)):
    dest = Path(UPLOAD_DIR) / file.filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        f.write(file.file.read())
    n = upsert_document(dest, source="upload")
    return {"ok": True, "chunks": n, "path": str(dest)}


@app.post("/reindex")
def reindex_api():
    root = Path(WATCH_DIR)
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
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    global collection
    collection = client.get_or_create_collection(COLLECTION)
    return {"ok": True}

@app.get("/ollama_health")
def ollama_health():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        ct = r.headers.get("content-type", "")
        body = r.json() if "application/json" in ct else r.text
        return {"ok": r.status_code == 200, "status": r.status_code, "body": body}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama unreachable: {e}")

@app.get("/embed_health")
def embed_health():
    try:
        vec = EMBEDDER.encode(["hello world"])[0]
        return {"ok": True, "dim": len(vec), "preview": vec[:8]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embed error: {e}")


# ---- folder watcher ----
class Handler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        p = Path(event.src_path)
        if p.suffix.lower() in SUPPORTED:
            try:
                upsert_document(p, source="watched")
            except Exception as e:
                print("ingest error:", e)

    def on_modified(self, event):
        self.on_created(event)


if Path(WATCH_DIR).exists():
    obs = Observer()
    obs.schedule(Handler(), WATCH_DIR, recursive=True)
    obs.start()


# ---------------- Ingest queue & jobs ----------------
INGEST_Q: "queue.Queue[tuple[str,str,str]]" = queue.Queue(maxsize=Q_MAX)
JOBS: dict[str, dict] = {}


def _ingest_worker():
    while True:
        job_id, path, source = INGEST_Q.get()
        try:
            t0 = time.time()
            chunks = upsert_document(Path(path), source=source)
            JOBS[job_id] = {"status": "done", "path": path, "chunks": chunks, "ms": int((time.time() - t0) * 1000)}
        except Exception as e:
            JOBS[job_id] = {"status": "error", "path": path, "error": str(e)}
        finally:
            INGEST_Q.task_done()


threading.Thread(target=_ingest_worker, daemon=True).start()


@app.post("/upload_async")
def upload_async_api(file: UploadFile = File(...)):
    dest = Path(UPLOAD_DIR) / file.filename
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
def _is_under_allowed(p: Path) -> bool:
    roots = [Path(UPLOAD_DIR).resolve(), Path(WATCH_DIR).resolve()]
    try:
        rp = p.resolve()
        for r in roots:
            try:
                if rp.is_relative_to(r.resolve()):
                    return True
            except AttributeError:
                # Python 3.9 doesn't have Path.is_relative_to
                if str(rp).startswith(str(r.resolve()) + os.sep):
                    return True
        return False
    except Exception:
        return False


@app.get("/download")
def download(path: str, inline: bool = False):
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "file not found")
    if not _is_under_allowed(p):
        raise HTTPException(403, "forbidden path")

    mt, _ = mimetypes.guess_type(str(p))
    resp = FileResponse(
        path=str(p),
        media_type=mt or "application/octet-stream",
        filename=p.name,
    )
    if inline:
        resp.headers["Content-Disposition"] = f'inline; filename="{p.name}"'
    return resp


class ZipRequest(BaseModel):
    paths: List[str]


@app.post("/download_zip")
def download_zip(req: ZipRequest):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for raw in req.paths or []:
            p = Path(raw)
            if not (p.exists() and p.is_file() and _is_under_allowed(p)):
                continue
            z.write(str(p), arcname=p.name)
    buf.seek(0)
    headers = {"Content-Disposition": 'attachment; filename="documents.zip"'}
    return StreamingResponse(buf, media_type="application/zip", headers=headers)


# ---------------- Listing (paged) ----------------
@app.get("/list_page")
def list_page(
    page: int = 1,
    size: int = 100,
    source: Optional[str] = Query(None, description="upload|watched"),
    prefix: Optional[str] = Query(None, description="path startswith filter"),
):
    data = collection.get(include=["metadatas"], limit=1_000_000)
    paths = []
    for m in data.get("metadatas", []):
        if not m:
            continue
        p = m.get("path")
        s = m.get("source")
        if not p:
            continue
        if source and s != source:
            continue
        if prefix and not str(p).startswith(prefix):
            continue
        paths.append(p)
    unique = sorted(set(paths))
    total = len(unique)

    page = max(page, 1)
    size = max(min(size, 1000), 1)
    start = (page - 1) * size
    end = min(start + size, total)
    items = unique[start:end]

    return {
        "page": page,
        "size": size,
        "total": total,
        "has_next": end < total,
        "next_page": page + 1 if end < total else None,
        "items": items,
    }


@app.get("/list_details_page")
def list_details_page(
    page: int = 1,
    size: int = 100,
    source: Optional[str] = Query(None),
    prefix: Optional[str] = Query(None),
):
    data = collection.get(include=["ids", "metadatas"], limit=1_000_000)
    by_path: Dict[str, Dict[str, Any]] = {}
    for _id, m in zip(data.get("ids", []), data.get("metadatas", [])):
        if not m:
            continue
        p = m.get("path")
        s = m.get("source")
        if not p:
            continue
        if source and s != source:
            continue
        if prefix and not str(p).startswith(prefix):
            continue
        entry = by_path.setdefault(p, {"path": p, "source": s, "chunks": 0})
        entry["chunks"] += 1

    docs = sorted(by_path.values(), key=lambda r: r["path"])
    total = len(docs)

    page = max(page, 1)
    size = max(min(size, 500), 1)
    start = (page - 1) * size
    end = min(start + size, total)
    items = docs[start:end]

    return {
        "page": page,
        "size": size,
        "total": total,
        "has_next": end < total,
        "next_page": page + 1 if end < total else None,
        "docs": items,
    }


# ---------------- Filtered search ----------------
def search_filtered(
    query: str,
    k: int = 4,
    path: Optional[str] = None,
    paths: Optional[List[str]] = None,
    prefix: Optional[str] = None,
    source: Optional[str] = None,
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
    elif source:
        res = collection.query(query_embeddings=[qemb], n_results=max(k, 12),
                               include=include, where={"source": source})
        results.extend(to_hits(res))
    else:
        res = collection.query(query_embeddings=[qemb], n_results=k, include=include)
        results.extend(to_hits(res))

    seen: Dict[Any, Dict[str, Any]] = {}
    for h in results:
        key = (h["meta"].get("path"), h["meta"].get("chunk"))
        if key not in seen or h["score"] > seen[key]["score"]:
            seen[key] = h
    hits = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:k]
    return hits


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

    hits = search_filtered(query, k=k, path=path, paths=paths, prefix=prefix, source=source)
    answer = ask_with_context(query, hits, chat_history=history, model=model)

    used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if n.isdigit()})
    filtered = [hits[i-1] for i in used if 1 <= i <= len(hits)] if used else []
    return {"answer": answer, "sources": filtered}
