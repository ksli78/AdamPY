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
import base64
import tempfile
from pathlib import Path as _Path
from typing import List, Optional, Dict, Any, Tuple

import requests
from fastapi import FastAPI, UploadFile, File, Query, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
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
        providers = ort.get_available_providers()
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
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "720m")
EMBED_MODEL_DIR = os.getenv("EMBED_MODEL_DIR", "/opt/adam/models/nomic-ai/nomic-embed-text")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3:8b")  # default to fast 8B
CHROMA_DIR = os.getenv("CHROMA_DIR", "/srv/rag/chroma")
WATCH_DIR = os.getenv("WATCH_DIR", "/srv/rag/watched")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/srv/rag/uploads")
COLLECTION = os.getenv("COLLECTION", "company_docs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))
Q_MAX = int(os.getenv("INGEST_QUEUE_MAX", "8"))

# ---------------- Model aliasing ----------------
ALIAS_MAP = {
    # Keep any local aliases you still use; avoid mapping removed 70B family
    "Adam Lite": "adam-lite:latest",
    "adam-lite": "adam-lite:latest",
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


def sha1(path: _Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path: _Path) -> str:
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


def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    import json
    safe: Dict[str, Any] = {}
    for k, v in (meta or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            safe[k] = v
        elif isinstance(v, (list, dict, tuple, set)):
            try:
                safe[k] = json.dumps(v, ensure_ascii=False)
            except Exception:
                safe[k] = str(v)
        else:
            safe[k] = str(v)
    return safe



# -------- New: upsert pre-parsed TEXT (SharePoint path) --------
def upsert_text(doc_id: str, text: str, base_meta: Dict[str, Any]) -> int:
    """Upsert pre-parsed TEXT (string) as chunks+embeddings, storing ONLY embeddings+metadata."""
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
        m = _sanitize_metadata(dict(base_meta))
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


def rerank_sources(question: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rerank retrieved chunks using Mistral based on relevance to the question."""
    if not chunks:
        return []
    top = chunks[:10]
    prompt_lines = [
        f"Given the question: {question}",
        "Rank the following snippets in order of relevance to the question.",
        "Respond with a comma-separated list of snippet numbers in ranked order.",
        "Snippets:"
    ]
    snippet_lines = []
    for i, ch in enumerate(top, 1):
        snippet = (ch.get("text") or "")[:200].replace("\n", " ")
        snippet_lines.append(f"{i}. {snippet}")
    user_prompt = "\n".join(prompt_lines + snippet_lines)

    messages = [
        {"role": "system", "content": "You rank text snippets by relevance."},
        {"role": "user", "content": user_prompt},
    ]
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": "mistral-7b-instruct",
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0},
                "keep_alive": OLLAMA_KEEP_ALIVE,
            },
            timeout=60,
        )
        if r.status_code == 200:
            text = (r.json().get("message") or {}).get("content", "")
            order = [int(n) for n in re.findall(r"\d+", text) if 1 <= int(n) <= len(top)]
            ranked = [top[i - 1] for i in order if 1 <= i <= len(top)]
            ranked += [top[i] for i in range(len(top)) if (i + 1) not in order]
            return ranked[:5]
    except Exception as e:
        print("Rerank failed:", e)
    return top[:5]


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
            json={
                "model": model_tag,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 1024, "top_p": 0.95},
                "keep_alive": OLLAMA_KEEP_ALIVE  # keep the 8B resident
            },
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


def filter_cited_sources(answer: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return only chunks whose text shares keywords with the answer and are cited."""
    if not answer or not chunks:
        return []
    used = {int(n) for n in re.findall(r"\[(\d+)\]", answer) if n.isdigit()}
    if not used:
        return []
    answer_words = set(re.findall(r"\b\w{4,}\b", answer.lower()))
    filtered = []
    for ch in chunks:
        idx = ch.get("index")
        if idx not in used:
            continue
        snippet_words = set(re.findall(r"\b\w{4,}\b", (ch.get("text") or "").lower()))
        if len(answer_words & snippet_words) >= 3:
            filtered.append(ch)
    return filtered

def rewrite_prompt(prompt: str) -> str:
    """Use Mistral to safely rewrite vague prompts, without changing intent."""
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that rewrites vague document questions into clearer, more precise ones "
                    "without changing their meaning. Do NOT guess or introduce new topics. "
                    "Only rewrite if the original query is unclear or incomplete."
                )
            },
            {
                "role": "user",
                "content": f"Original query: {prompt}\n\nRewritten query:"
            }
        ]

        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": "mistral-7b-instruct",
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.3},
                "keep_alive": OLLAMA_KEEP_ALIVE
            },
            timeout=30
        )

        if r.status_code == 200:
            j = r.json()
            rewritten = (j.get("message") or {}).get("content", "").strip()

            # Basic sanity check: don't allow rewrites that lose all original keywords
            if rewritten and any(word.lower() in rewritten.lower() for word in prompt.split()):
                return rewritten

        return prompt  # fallback if empty or invalid rewrite
    except Exception as e:
        print("Prompt rewrite failed:", e)
        return prompt

# ---- API models & endpoints ----
class QueryBody(BaseModel):
    query: str
    k: int = 4
    history: Optional[List[dict]] = None
    model: Optional[str] = None  # "Adam Large", "Adam Lite", or raw Ollama tag
    # New optional filters (non-breaking)
    org: Optional[str] = None
    category: Optional[str] = None
    doc_code: Optional[str] = None
    owner: Optional[str] = None
    rewrite: bool = True


class QueryResponse(BaseModel):
    """Response schema for the /query endpoint."""
    answer: str
    sources: List[Dict[str, Any]]
    original_query: str
    rewritten_query: str
    prompt_rewritten: bool


@app.post("/query", response_model=QueryResponse)
def query_api(body: QueryBody) -> QueryResponse:
    if body.rewrite:
        rewritten_query_raw = rewrite_prompt(body.query).strip()
        m = re.search(r"Rewritten Query:\s*(.*)", rewritten_query_raw, re.DOTALL)
        if m:
            rewritten_query = m.group(1).strip()
        else:
            rewritten_query = next(
                (ln.strip() for ln in rewritten_query_raw.splitlines() if ln.strip()),
                body.query,
            )
        prompt_rewritten = rewritten_query != body.query
    else:
        rewritten_query = body.query
        prompt_rewritten = False

    # Use filtered search if filters provided; else regular search
    if body.org or body.category or body.doc_code or body.owner:
        hits = search_filtered(rewritten_query, k=10,
                               source=None, org=body.org, category=body.category,
                               doc_code=body.doc_code, owner=body.owner)
    else:
        hits = search(rewritten_query, k=10)
    hits = rerank_sources(rewritten_query, hits)
    for idx, h in enumerate(hits, 1):
        h["index"] = idx
    answer = ask_with_context(rewritten_query, hits, chat_history=body.history, model=body.model)
    relevant = filter_cited_sources(answer, hits)
    rich = []
    for h in relevant:
        meta = h.get("meta", {}) or {}
        rich.append({
            "index": h.get("index"),
            "title": meta.get("title"),
            "org": meta.get("org"),
            "category": meta.get("category"),
            "version": meta.get("version"),
            "revision_date": meta.get("revision_date"),
            "doc_code": meta.get("doc_code"),
            "sp_web_url": meta.get("sp_web_url"),
            "path": meta.get("path"),   # present for legacy local docs
            "score": h.get("score"),
            "snippet": (h.get("text") or "")[:400],
        })
    return QueryResponse(
        answer=answer,
        sources=rich,
        original_query=body.query,
        rewritten_query=rewritten_query,
        prompt_rewritten=prompt_rewritten,
    )


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


@app.get("/list_docs")
def list_documents():
    try:
        collection = collection = client.get_or_create_collection(COLLECTION)

        # Fetch all document entries with metadata
        results = collection.get(include=["metadatas", "documents"], limit=10000)

        docs = []
        for i in range(len(results["ids"])):
            doc_info = {
                "doc_id": results["ids"][i],
                "metadata": results["metadatas"][i],
            }
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
JOBS: dict[str, dict] = {}


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


threading.Thread(target=_ingest_worker, daemon=True).start()


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
                # Python 3.9 doesn't have Path.is_relative_to
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
            p = _Path(raw)
            if not (p.exists() and p.is_file() and _is_under_allowed(p)):
                continue
            z.write(str(p), arcname=p.name)
    buf.seek(0)
    headers = {"Content-Disposition": 'attachment; filename="documents.zip"'}
    return StreamingResponse(buf, media_type="application/zip", headers=headers)


# ---------------- Filtered search ----------------
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

    hits = search_filtered(query, k=10, path=path, paths=paths, prefix=prefix, source=source,
                           org=org, category=category, doc_code=doc_code, owner=owner)
    hits = rerank_sources(query, hits)
    for idx, h in enumerate(hits, 1):
        h["index"] = idx
    answer = ask_with_context(query, hits, chat_history=history, model=model)
    filtered = filter_cited_sources(answer, hits)
    return {"answer": answer, "sources": filtered}


# ---------------- New: Pydantic model for SharePoint ingest ----------------
class IngestDocument(BaseModel):
    # Identity & links (from SharePoint)
    sp_site_id: Optional[str] = None
    sp_list_id: Optional[str] = None
    sp_item_id: Optional[str] = None
    sp_drive_id: Optional[str] = None
    sp_file_id: Optional[str] = None
    sp_web_url: str

    # Versioning
    etag: Optional[str] = None
    version_label: Optional[str] = None

    # Core metadata
    title: Optional[str] = None
    doc_code: Optional[str] = None
    org_code: Optional[str] = None
    org: Optional[str] = None
    category: Optional[str] = None
    owner: Optional[str] = None
    version: Optional[str] = None
    revision_date: Optional[str] = None
    latest_review_date: Optional[str] = None
    document_review_date: Optional[str] = None
    review_approval_date: Optional[str] = None
    keywords: Optional[List[str]] = None
    enterprise_keywords: Optional[List[str]] = None
    association_ids: Optional[List[str]] = None
    domain: Optional[str] = "HR"
    allowed_groups: Optional[List[str]] = None

    # Content options (you’ll use content_bytes)
    file_name: Optional[str] = None
    content_bytes: Optional[str] = None   # base64 of the file bytes
    text_content: Optional[str] = None    # if you pre-extract text in your SP worker

    # Optional overrides
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    persist: Optional[bool] = False       # ignored; never persist files locally


# ---------------- New: SharePoint-first ingest (no local persistence) ----------------
@app.post("/ingest_document")
def ingest_document_api(body: IngestDocument):
    """
    Accept SharePoint metadata + content (base64 bytes or pre-extracted text),
    parse to text in-memory (if needed), chunk, embed, upsert. No file persistence.
    """
    # Use per-request chunk overrides if provided
    global CHUNK_SIZE, CHUNK_OVERLAP
    orig_size, orig_overlap = CHUNK_SIZE, CHUNK_OVERLAP
    try:
        if body.chunk_size: CHUNK_SIZE = int(body.chunk_size)
        if body.chunk_overlap: CHUNK_OVERLAP = int(body.chunk_overlap)

        # Build stable versioned doc_id
        version_key = body.version_label or body.etag or "v1"
        spid = str(body.sp_item_id or body.sp_file_id or body.sp_web_url)
        doc_id = f"{spid}:{version_key}"

        # Base metadata attached to every chunk
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

        # 1) If text provided, use directly (fastest)
        if body.text_content and str(body.text_content).strip():
            n = upsert_text(doc_id, body.text_content, base_meta)
            return {"ok": True, "doc_id": doc_id, "chunks": n, "used": "text_content"}

        # 2) Else parse from base64 content bytes ephemerally
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
        CHUNK_SIZE, CHUNK_OVERLAP = orig_size, orig_overlap


# ---------------- New: Delete by SharePoint identity ----------------
@app.delete("/delete_sp")
def delete_by_sp(sp_item_id: Optional[str] = Query(None),
                 sp_file_id: Optional[str] = Query(None),
                 version: Optional[str] = Query(None),
                 etag: Optional[str] = Query(None)):
    """
    Delete all chunks for a SharePoint document identity. If version/etag is
    supplied, only that version is removed; otherwise all versions are removed.
    """
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
