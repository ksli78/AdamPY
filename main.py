# --- import path shim (ensures /srv/rag/app is importable even under systemd) ---
import os, sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # /srv/rag/app
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --- end shim ---


# If someone runs the app as a package (e.g., app.main:app), this fallback helps
from config import settings  # type: ignore

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
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import deque
from pathlib import Path as _Path
from typing import List, Optional, Dict, Any, Tuple, Set



import logging
from systemd import journal

logger = logging.getLogger('rag')
logger.setLevel(logging.DEBUG)  # or INFO, WARNING, etc.

journal_handler = journal.JournalHandler()
journal_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(levelname)s: %(message)s')
journal_handler.setFormatter(formatter)

logger.addHandler(journal_handler)

_DEFAULT_CATEGORY_VOCAB = {
    # add SharePoint functional areas:
    "contract management", "employee resources",
    "management resources", "product support",
    "quality & compliance", "science",
    "information technology", "operations & test",
    "business office", "cptio", "opm",
    "performance office", "pmdo",
    "safety & mission assurance", "security"
}
# rule-based keyword nudges (helps separate close siblings)
# Rationale:
# - CLG-EN-PO-0301 (Work Hours…): mentions Decisions + "Time Off Request" (PTO workflow). 
# - EN-PO-0276 (Personal Leave): emphasizes unpaid leave mechanics and J375 form.
# These anchors are strong disambiguators for PTO vs. Personal Leave. 
# (Adjust to your domain as you add more policies.)
_RULES = [
    {
        "pattern": r"\b(time off request)\b",  # no bare “decisions”
        "add_keywords": ["PTO","Time Off Request","Portal->Tools->Decisions"],
        "force_category": "policy"
    },
    {
        "pattern": r"\bJ375\b",
        "add_keywords": ["personal leave", "leave of absence", "unpaid leave", "J375 form"],
        # keep category from LLM, just add keywords
    },
]
try:
    import requests
except ModuleNotFoundError:  # pragma: no cover - fallback to urllib
    from urllib import request as urlrequest

    class _Response:
        def __init__(self, resp):
            self.status_code = resp.status
            self.headers = resp.headers
            self._body = resp.read()

        @property
        def text(self):
            return self._body.decode("utf-8")

        def json(self):
            return json.loads(self.text)

    class requests:  # type: ignore
        @staticmethod
        def post(url, json=None, timeout=None):
            data = json.dumps(json).encode("utf-8") if json is not None else None
            req = urlrequest.Request(url, data=data, headers={"Content-Type": "application/json"})
            resp = urlrequest.urlopen(req, timeout=timeout)
            return _Response(resp)

        @staticmethod
        def get(url, timeout=None):
            req = urlrequest.Request(url)
            resp = urlrequest.urlopen(req, timeout=timeout)
            return _Response(resp)
from fastapi import FastAPI, UploadFile, File, Query, Body, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

# ---------------- Vector DB ----------------
import chromadb

try:  # Prefer local package name but support being nested under ``app``
    from adampy.services.ollama_client import OllamaClient
    from adampy.pipeline.semantic_rag import (
        generate_query_variants,
        dense_retrieve,
        rrf_fuse,
        load_reranker_or_reuse,
        rerank,
        build_grounded_answer,
    )
    from adampy.pipeline.citations import validate_and_fix_citations
    from adampy.services.search import search_filtered
except ModuleNotFoundError:  # pragma: no cover
    from app.adampy.services.ollama_client import OllamaClient
    from app.adampy.pipeline.semantic_rag import (
        generate_query_variants,
        dense_retrieve,
        rrf_fuse,
        load_reranker_or_reuse,
        rerank,
        build_grounded_answer,
    )
    from app.adampy.pipeline.citations import validate_and_fix_citations
    from app.adampy.services.search import search_filtered

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
import difflib

# ---------------- Embeddings: nomic-embed-text via ONNXRuntime ----------------
from adampy.services.embedding import NomicOnnxEmbedder, ChromaEmbedder

def resolve_model(name: Optional[str]) -> str:
    """Return an Ollama model tag given a friendly name or raw tag."""
    if not name:
        return settings.CHAT_MODEL
    return settings.ALIAS_MAP.get(name, name)


# Initialize embedder early so startup fails fast if model missing
EMBEDDER = NomicOnnxEmbedder(settings.EMBED_MODEL_DIR)


CHROMA_EMBED = ChromaEmbedder(EMBEDDER)


def embed(texts: List[str]) -> List[List[float]]:
    return CHROMA_EMBED.embed_documents(texts)


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

client = chromadb.PersistentClient(path=settings.CHROMA_DIR)

class InferMetadataRequest(BaseModel):
    text: str
    # Optional hints (from your crawler)
    title: Optional[str] = None
    doc_code: Optional[str] = None
    category_vocab: Optional[List[str]] = None  # override/extend defaults
    max_keywords: int = 12                       # cap the list size
    lower_keywords: bool = False                 # keep casing by default
    dedupe_keywords: bool = True
    apply_rules: bool = True                     # apply _RULES anchors (Decisions/J375) for now

class InferMetadataResponse(BaseModel):
    summary: str
    category: str
    keywords: List[str]
    debug: Dict[str, List[str]] = {}             # shows what rules matched / changes applied

def _normalize_category(raw: str, vocab: set[str]) -> str:
    c = (raw or "").strip().lower()
    if not c:
        return "policy" if "policy" in vocab else (sorted(vocab)[:1] or ["misc"])[0]
    # single- or two-word; map common variants
    c = re.sub(r"[^a-z0-9 ]+", "", c)
    aliases = {
       # existing mappings…
       "human resources|hr policy|hr": "hr",
       "information technology|it policy|it": "information technology",
       "engineering department|engineering|jets|flight lab": "engineering",
       "analysis|assembly and fabrication|certification and acceptance|configuration management|design|software engineering|systems engineering|guidance navigation & control|operations and integration|project risk management|verification and validation|eee parts": "engineering",
       "manufacturing|procurements|subcontracts|procurement|shipping and receiving": "product support",
       "goals|innovations|jets management|management system|to management|security": "contract management",
       "career development|employee information|onboarding|timekeeping|training": "employee resources",
       "requisitions|resource management|telecommuting": "management resources",
       "facility safety|flight safety|ppe|safety management": "safety",
       "facility maintenance|facility management|pressure systems|property management|tspf ms matrix": "facilities",
       "quality assurance|quality control|gidep": "quality & compliance",
       "science|research": "science",
       "it policies and procedures|it policies": "information technology"
       # add any other relevant patterns from your SharePoint taxonomy
    }
    for patt, val in aliases.items():
        if re.fullmatch(patt, c):
            c = val
            break
    return c if c in vocab else (min(vocab, key=lambda v: len(v)) if vocab else c or "misc")

def _clean_keywords(kw: List[str], *, max_n: int, lower: bool, dedupe: bool) -> List[str]:
    out: List[str] = []
    seen = set()
    for k in kw or []:
        s = (k or "").strip()
        if not s:
            continue
        # strip quotes/brackets and punctuation padding
        s = s.strip(" \t\r\n\"'[](){}")
        if lower:
            s = s.lower()
        if dedupe:
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
        out.append(s)
        if len(out) >= max_n:
            break
    return out

def _apply_anchor_rules(text: str, existing: List[str], vocab: set[str], apply: bool) -> tuple[List[str], List[str]]:
    """Return (keywords, hits): add keywords and maybe force category based on rule hits."""
    hits: List[str] = []
    kw = list(existing)
    if not apply:
        return kw, hits
    t = text or ""
    for rule in _RULES:
        if re.search(rule["pattern"], t, flags=re.I):
            hits.append(rule["pattern"])
            for k in rule.get("add_keywords", []):
                if k not in kw:
                    kw.append(k)
    return kw, hits

def _ensure_collection(name: Optional[str] = None):
    """Return a Chroma collection using our embedder, recreating if mismatched.

    If the on-disk collection was created with a different embedding dimension,
    Chroma raises a ValueError. In that case, delete and recreate the specific
    collection name so callers always get a usable handle.
    """
    col_name = (name or settings.COLLECTION).strip()
    try:
        return client.get_or_create_collection(col_name, embedding_function=CHROMA_EMBED)
    except ValueError:
        # Existing collection has conflicting embedding function; reset it.
        try:
            client.delete_collection(col_name)
        except Exception:
            pass
        return client.get_or_create_collection(col_name, embedding_function=CHROMA_EMBED)


collection = _ensure_collection()
retriever = collection

_debug_lock = threading.Lock()
_debug_buffer = deque(maxlen=10)
_DEBUG_API_KEY = os.getenv("DEBUG_API_KEY", "")


def _now_ms():
    return int(time.time() * 1000)


def _truncate(s: str, n: int = 280) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= n else s[:n] + "\u2026"


def _get_text_for_hit(hit: dict) -> str:
    """
    Return usable passage text for a retrieved hit.
    Tries multiple common keys (on the hit and in meta/metadata) and falls back.
    """

    logger.debug("Received dict: %s", json.dumps(hit, indent=2))

    for k in ("page_content", "content", "text", "body", "snippet", "text_content"):
        v = hit.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    meta = hit.get("meta") or hit.get("metadata") or {}
    for k in ("page_content", "content", "text", "body", "snippet", "text_content"):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    title = (hit.get("title") or meta.get("title") or "").strip()
    url = (hit.get("sp_web_url") or meta.get("sp_web_url") or hit.get("path") or meta.get("path") or "").strip()
    if title or url:
        return f"{title}\n{url}".strip()

    return ""

# ---- BGE reranker (local/offline) with safe fallback ----
_BGE_PATH = settings.RERANKER_MODEL_PATH
_BGE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_BGE_OK = False
try:
    _BGE_TOK = AutoTokenizer.from_pretrained(_BGE_PATH, local_files_only=True)
    _BGE_MODEL = AutoModelForSequenceClassification.from_pretrained(_BGE_PATH, local_files_only=True)
    _BGE_MODEL.to(_BGE_DEVICE)
    _BGE_MODEL.eval()
    _BGE_OK = True
except Exception:
    _BGE_OK = False


def _bge_scores(query: str, texts: List[str], batch_size: int = 16, max_length: int = 512) -> List[float]:
    scores: List[float] = []
    for i in range(0, len(texts), batch_size):
        bt = texts[i:i+batch_size]
        enc = _BGE_TOK(text=[query]*len(bt), text_pair=bt, truncation=True,
                       max_length=max_length, padding=True, return_tensors="pt")
        enc = {k: v.to(_BGE_DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            logits = _BGE_MODEL(**enc).logits.squeeze(-1)  # [B]
        scores.extend(torch.sigmoid(logits).detach().cpu().tolist())
    return scores


_TOKEN_RE = re.compile(r"\b[\w:.-]{3,}\b", re.UNICODE)


def _tokens_generic(s: str) -> set:
    return set(t.lower() for t in _TOKEN_RE.findall(s or ""))

def _extract_passage(text: str, query: str, window_chars: int = 800) -> str:
    """
    Generic passage picker: prefer segments whose tokens overlap with the query.
    No domain-specific words; purely lexical overlap + small window.
    """
    t = (text or "").strip()
    if not t:
        return ""
    qtok = _tokens_generic(query)
    if not qtok:
        return t[:window_chars]

    # split to sentences (very rough)
    sents = re.split(r'(?<=[.!?])\s+', t)
    # score each sentence by token overlap
    scored = [(i, len(_tokens_generic(s) & qtok), s) for i, s in enumerate(sents)]
    scored.sort(key=lambda x: x[1], reverse=True)
    if not scored or scored[0][1] == 0:
        return t[:window_chars]

    # take a window around the best sentence
    i_best = scored[0][0]
    left = max(0, i_best - 2)
    right = min(len(sents), i_best + 3)
    excerpt = " ".join(sents[left:right]).strip()
    if len(excerpt) < window_chars // 2 and len(t) > window_chars:
        # pad to window size if too short
        start = max(0, t.lower().find(excerpt.lower()) - (window_chars // 4))
        return t[start:start + window_chars]
    return excerpt[:window_chars]

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

def chunk_text(text: str, size: int = settings.CHUNK_SIZE, overlap: int = settings.CHUNK_OVERLAP) -> List[str]:
    text = text.strip()
    if not text:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


HEADER_FOOTER_RE = re.compile(r'^(Page\b|Revision\b)', re.IGNORECASE)
SIGNATURE_RE = re.compile(r'\n(?:Regards|Sincerely|Thank you|Thanks|Best)[\s\S]*$', re.IGNORECASE)
LEGAL_RE = re.compile(r'\n(?:Confidentiality Notice|DISCLAIMER:)[\s\S]*$', re.IGNORECASE)

def clean_document_text(text: str) -> str:
    """Basic cleanup: remove headers/footers, signatures, and normalize whitespace."""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = []
    for ln in text.split('\n'):
        st = ln.strip()
        if HEADER_FOOTER_RE.match(st):
            continue
        lines.append(st)
    cleaned = '\n'.join(lines)
    cleaned = SIGNATURE_RE.sub('', cleaned)
    cleaned = LEGAL_RE.sub('', cleaned)
    cleaned = re.sub(r'\n{2,}', '\n', cleaned)
    cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)
    return cleaned.strip()

def call_llm(prompt: str) -> str:
    """Send a prompt to the configured LLM and return the raw text response."""
    try:
        r = requests.post(
f"{settings.OLLAMA_URL}/api/chat",
            json={
"model": resolve_model(settings.SUMMARY_MODEL),
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0},
"keep_alive": settings.OLLAMA_KEEP_ALIVE,
            },
            timeout=120,
        )
        if r.status_code == 200:
            return (r.json().get("message") or {}).get("content", "")
        else:
            print(f"Warning: call_llm HTTP {r.status_code}: {r.text}")
    except Exception as e:
        print(f"Warning: call_llm request failed: {e}")
    return ""


def summarize_document(text_content):
    prompt = f"""
    Summarize the following document content and extract metadata. 
    Respond only with JSON using this exact format:

    {{
      "summary": "<brief summary of the content>",
      "category": "<one or two word category>",
      "keywords": ["<keyword1>", "<keyword2>", ...]
    }}

    Document:
    {text_content}

    Only respond with valid JSON. Do not include any explanatory text or commentary.
    """

    raw_content = call_llm(prompt)
    print("LLM raw content:", raw_content)

    # Try full JSON parse
    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError:
        match = re.search(r'\{.*?\}', raw_content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except:
                print("Still failed to parse trimmed JSON.")
                data = {}
        else:
            print("Regex match for JSON failed.")
            data = {}

    summary = data.get("summary", "").strip()
    category = data.get("category", "").strip()
    keywords = data.get("keywords", []) if isinstance(data.get("keywords", []), list) else []

    print(f"Parsed summary: {summary}")
    print(f"Parsed category: {category}")
    print(f"Parsed keywords: {keywords}")

    return summary, category, keywords


def upsert_document(path: _Path, source: str) -> int:
    text = read_text(path)
    if not text:
        return 0
    text = clean_document_text(text)
    if len(text) < 500:
        return 0
    summary, category, keywords = summarize_document(text)
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
    base_meta = {
        "source": source,
        "path": str(path),
        "summary": summary,
        "category": category,
        "keywords": keywords,
    }
    metas = []
    for i in range(len(chunks)):
        m = _sanitize_metadata(dict(base_meta))
        m["chunk"] = i
        metas.append(m)
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
    text = clean_document_text(text or "")
    if len(text) < 500:
        raise ValueError("document under 500 characters after cleanup")

    try:
        collection.delete(where={"doc_id": doc_id})
    except Exception:
        pass
    chunks = chunk_text(text)
    if not chunks:
        return 0
    embs = embed(chunks)
    ids = [f"{doc_id}:{i}" for i in range(len(chunks))]

    # Ensure LLM metadata fields are strings / JSON-serializable
    meta = dict(base_meta or {})
    if not meta.get("summary") and not meta.get("category") and not meta.get("keywords"):
        s, c, k = summarize_document(text)
        meta.update({"summary": s, "category": c, "keywords": k})

    summary = meta.get("summary", "")
    category = meta.get("category", "")
    keywords = meta.get("keywords", [])

    if not isinstance(summary, str):
        try:
            summary = json.dumps(summary, ensure_ascii=False)
        except Exception:
            summary = str(summary)
    if not isinstance(category, str):
        category = str(category)
    if not isinstance(keywords, str):
        try:
            keywords = json.dumps(keywords, ensure_ascii=False)
        except Exception:
            keywords = str(keywords)

    meta.update({"summary": summary, "category": category, "keywords": keywords})
    print(f"upsert_text metadata summary={summary!r}, category={category!r}, keywords={keywords}")

    metas = []
    for i in range(len(chunks)):
        m = dict(meta)
        m["doc_id"] = doc_id
        m["chunk"] = i
        m = _sanitize_metadata(m)
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


def hybrid_rerank(query: str, retriever, reranker_model_name: str,
                  initial_k: int = 15, final_k: int = 5,
                  where: Optional[Dict[str, Any]] = None,
                  debug: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Retrieve top N chunks via Chroma then rerank them with a local BGE model."""
    start_retrieve = _now_ms()
    qemb = embed([query])[0]
    query_kwargs = {
        "query_embeddings": [qemb],
        "n_results": initial_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where
    res = retriever.query(**query_kwargs)
    retrieve_ms = _now_ms() - start_retrieve
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    chunks = [{"text": d, "meta": m, "score": float(1.0 / (1e-5 + dist))}
              for d, m, dist in zip(docs, metas, dists)]
    if not chunks:
        if debug is not None:
            debug.update({
                "pre_llm_candidates": [],
                "doc_code_boosts": [],
                "reranker": {"raw_text": "", "parsed": [], "threshold": 0.0, "fallback_used": True},
                "post_rerank_hits": [],
                "timing_ms": {"retrieve": retrieve_ms, "rerank": 0},
            })
        return []

    doc_boosts: List[Dict[str, Any]] = []
    _apply_metadata_boost(query, chunks, doc_boosts)

    chunks.sort(key=lambda x: x["score"], reverse=True)
    pre_llm = [dict(ch) for ch in chunks]
    if debug is not None:
        debug["pre_llm_candidates"] = []
        for i, ch in enumerate(pre_llm, 1):
            meta = ch.get("meta") or {}
            debug["pre_llm_candidates"].append({
                "index": i,
                "score_pre": ch.get("score"),
                "doc_code": meta.get("doc_code"),
                "title": meta.get("title"),
                "category": meta.get("category"),
                "revision_date": meta.get("revision_date"),
                "sp_web_url": meta.get("sp_web_url"),
                "snippet": _truncate(ch.get("text") or ch.get("snippet"), 280),
            })
        debug["doc_code_boosts"] = doc_boosts

    # --- BGE rerank attempt over a widened pool ---
    pool = chunks[:max(len(chunks), 40)]  # widen pool (generic)
    texts = [(ch.get("text") or "")[:4000] for ch in pool]

    start_rerank = _now_ms()
    rerank_fallback_used = False
    if _BGE_OK and texts:
        try:
            bge = _bge_scores(query, texts, batch_size=16, max_length=512)
            for ch, s in zip(pool, bge):
                ch["score"] = float(s) * ch.get("_boost", 1.0)
            pool.sort(key=lambda x: x["score"], reverse=True)
            chunks = pool
        except Exception:
            rerank_fallback_used = True
    else:
        rerank_fallback_used = True

    if rerank_fallback_used:
        pass  # fallback to existing order/logic

    result = chunks[:final_k] if chunks else []
    rerank_ms = _now_ms() - start_rerank

    if debug is not None:
        debug["reranker"] = debug.get("reranker", {})
        debug["reranker"].update({
            "raw_text": debug["reranker"].get("raw_text", ""),
            "parsed": [{"index": i + 1, "score": h.get("score")} for i, h in enumerate(result)],
            "threshold": debug["reranker"].get("threshold", 0.0),
            "fallback_used": bool(rerank_fallback_used),
        })
        debug["post_rerank_hits"] = []
        for i, h in enumerate(result, 1):
            meta = h.get("meta") or {}
            debug["post_rerank_hits"].append({
                "index": i,
                "score": h.get("score"),
                "doc_code": meta.get("doc_code"),
                "title": meta.get("title"),
                "snippet": _truncate(h.get("text") or h.get("snippet"), 280),
            })
        debug["timing_ms"] = {"retrieve": retrieve_ms, "rerank": rerank_ms}

    return result

def rerank_sources(question: str, chunks: List[Dict[str, Any]],
                   debug: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Rerank retrieved chunks using local BGE scoring."""
    if not chunks:
        if debug is not None:
            debug["reranker"] = {"raw_text": "", "parsed": [], "threshold": 0.0, "fallback_used": True}
        return []

    doc_boosts: List[Dict[str, Any]] = []
    _apply_metadata_boost(question, chunks, doc_boosts)
    texts = []
    for ch in chunks:
        t = _get_text_for_hit(ch)
        texts.append(t[:4000])
    fallback_used = False
    if _BGE_OK:
        try:
            scores = _bge_scores(question, texts)
            for ch, s in zip(chunks, scores):
                ch["score"] = float(s) * ch.get("_boost", 1.0)
        except Exception as e:
            print("Rerank failed:", e)
            fallback_used = True
            for ch in chunks:
                ch["score"] = ch.get("score", 0.0)
    else:
        fallback_used = True

    chunks.sort(key=lambda x: x["score"], reverse=True)
    top = chunks[:3]

    if debug is not None:
        debug.update({
            "doc_code_boosts": doc_boosts,
            "reranker": {
                "raw_text": "",
                "parsed": [{"index": i + 1, "score": h.get("score")} for i, h in enumerate(top)],
                "threshold": 0.0,
                "fallback_used": fallback_used,
            },
        })
    return top

def _dehedge(text: str) -> str:
    patterns = [
        r'(?i)^\s*(according to|based on|from)\s+(the\s+)?(provided|given)\s+(context|information|documents)\s*[:,\-]*\s*',
        r'(?i)^\s*(according to|based on)\s+(the\s+)?document(s)?\s*[:,\-]*\s*',
    ]
    for p in patterns:
        text = re.sub(p, "", text, count=1)
    return text.strip()

def ask_with_context(question: str, hits: List[dict], chat_history: Optional[List[dict]] = None,
                     model: Optional[str] = None, force_citations: bool = False,
                     extra_system_prompt: str = "") -> str:
    ql = (question or "").lower()

    meta_triggers = [
        "your name", "what is your name", "what's your name",
        "who are you", "who is adam", "what does adam stand for",
        "what are you", "introduce yourself"
    ]
    if any(t in ql for t in meta_triggers):
        return "I am Adam - the Amentum Document and Assistance Model (ADAM)."

    context = "\n\n".join([f"[{h.get('index', i+1)}] {h['text']}" for i, h in enumerate(hits)])

    sys_prompt = (
        "You are Adam — the Amentum Document and Assistance Model (ADAM). "
        "Answer directly and succinctly. Do not start with phrases like "
        "'According to the provided context'. Use ONLY the provided context for factual claims and insert "
        "inline bracket citations like [1], [2] right after the sentence they support. "
        "Do not append a 'Sources:' section. If the answer is not in the context, say you do not know. "
        "CITATION RULES: Use only numeric bracket citations that correspond to the provided context blocks, e.g., [1], [2]. "
        "Do not use section numbers like [4.1], ranges like [1-3], or textual citations. Put the citation immediately after each claim it supports. "
        "COMPLETENESS RULES: If the user asks about definitions, boundaries, windows, or procedures, include all relevant elements present in the context (e.g., start and end times, total hours, tool/system names). "
        "NO FABRICATION: If a claim cannot be supported with a bracket citation from the context, say you do not know. "
        "OUTPUT FORMAT (HTML ONLY): Respond with a well-formed HTML fragment (not a full <html> page). Use <p> for paragraphs (each sentence starts with a capital letter). <ul> / <ol> with <li> for lists of steps or bullets. <table><thead>…</thead><tbody>…</tbody></table> for side-by-side facts. <strong>, <em>, <code>, <sup> as needed. CITATIONS: Put bracket citations inline at the end of the clause they support as <sup>[n]</sup>. Only use numbers that map to the provided context blocks. STYLE & SAFETY: Do not include <script>, inline CSS, external images, or arbitrary attributes. No markdown; HTML only. COMPLETENESS: For definition/boundary/steps questions, include all relevant elements present in the context (e.g., start and end times, total hours, and system/tool names). "
        "Direct answer in one sentence. One short follow-up sentence with any missing critical detail (e.g., 'begins Friday 12:00 noon and ends next Friday 11:59 a.m.'). Include the bracket citations inline."
    )
    if force_citations:
        sys_prompt += " You MUST include at least one citation if you answer. If unsure, say you do not know."
    if extra_system_prompt:
        sys_prompt += " " + extra_system_prompt

    messages = [{"role": "system", "content": sys_prompt}]
    if chat_history:
        messages.extend(chat_history)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    messages.append({"role": "user", "content": prompt})

    model_tag = resolve_model(model)

    try:
        r = requests.post(
f"{settings.OLLAMA_URL}/api/chat",
            json={
                "model": model_tag,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 1024, "top_p": 0.95},
"keep_alive": settings.OLLAMA_KEEP_ALIVE  # keep the 8B resident
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
    """Return chunks that are explicitly cited in the answer."""
    if not answer or not chunks:
        return []
    used = {int(n) for n in re.findall(r"\[(\d+)\]", answer) if n.isdigit()}
    if not used:
        return []
    ordered = []
    for ch in chunks:
        if ch.get("index") in used:
            ordered.append(ch)
    return ordered


def _extract_numeric_citations(answer: str, max_index: int) -> Tuple[Set[int], bool]:
    brackets = re.findall(r"\[([^\]]+)\]", answer or "")
    used: Set[int] = set()
    valid = True
    for b in brackets:
        if not b.isdigit():
            valid = False
            continue
        idx = int(b)
        if idx < 1 or idx > max_index:
            valid = False
        else:
            used.add(idx)
    return used, valid


def _ensure_html(text: str) -> Tuple[str, bool, List[str]]:
    tags = [t.lower() for t in re.findall(r"<\s*([a-zA-Z0-9]+)", text or "")]
    is_html = bool(tags)
    if not any(t in ("p", "ul", "ol", "table") for t in tags):
        t = (text or "").strip()
        if t:
            t = t[0].upper() + t[1:]
        text = f"<p>{t}</p>"
        tags = ["p"]
        is_html = True
    tag_summary = sorted({t for t in tags})
    return text, is_html, tag_summary


def _ensure_doc_code(meta: Dict[str, Any]) -> None:
    if meta.get("doc_code"):
        return
    url = meta.get("sp_web_url") or meta.get("path") or ""
    m = re.search(r"([A-Za-z]{2,}-[A-Za-z]{2,}-[A-Za-z]{2,}-\d{3,5})", url)
    if m:
        meta["doc_code"] = m.group(1)


def _apply_metadata_boost(query: str, chunks: List[Dict[str, Any]], doc_boosts: Optional[List[Dict[str, Any]]] = None) -> None:
    def _tokens(s: str) -> List[str]:
        return re.findall(r"\b\w+\b", (s or "").lower())

    q_tokens = set(_tokens(query))
    code_pat = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+")
    query_codes = {c.lower() for c in code_pat.findall(query)}
    if doc_boosts is None:
        doc_boosts = []
    for idx, ch in enumerate(chunks, 1):
        meta = ch.get("meta") or {}
        _ensure_doc_code(meta)
        fields = []
        cat = meta.get("category")
        if cat:
            fields.append(cat)
        title = meta.get("title")
        if title:
            fields.append(title)
        kws = meta.get("keywords")
        if kws:
            if isinstance(kws, list):
                fields.extend(kws)
            else:
                fields.append(kws)
        doc_code = meta.get("doc_code")
        if doc_code:
            fields.append(doc_code)
        version = meta.get("version")
        if version:
            fields.append(str(version))
        rev_date = meta.get("revision_date")
        if rev_date:
            fields.append(str(rev_date))

        meta_tokens = set()
        for f in fields:
            meta_tokens.update(_tokens(str(f)))

        matches = 0
        for qt in q_tokens:
            if qt in meta_tokens:
                matches += 1
            else:
                for mt in meta_tokens:
                    if difflib.SequenceMatcher(None, qt, mt).ratio() >= 0.8:
                        matches += 1
                        break

        boost = 1.0
        if matches:
            boost *= 1 + 0.1 * matches
        if doc_code:
            dc = str(doc_code).lower()
            for qc in query_codes:
                if dc == qc:
                    boost *= 3.0
                    doc_boosts.append({"index": idx, "reason": "exact", "multiplier": 3.0})
                    break
                norm_dc = dc.replace("-", "")
                norm_qc = qc.replace("-", "")
                if norm_dc == norm_qc or difflib.SequenceMatcher(None, norm_dc, norm_qc).ratio() >= 0.9:
                    boost *= 2.0
                    doc_boosts.append({"index": idx, "reason": "fuzzy", "multiplier": 2.0})
                    break
        if (meta.get("category") or "").lower() == "policy":
            boost *= 1.15
        ch.setdefault("_boost", boost)
        ch["score"] *= boost

def rewrite_prompt(prompt: str) -> str:
    """Use Mistral to safely rewrite vague prompts, without changing intent."""
    try:
        code_pat = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+")
        codes = code_pat.findall(prompt)

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
                "content": f"Original query: {prompt}\n\nRewritten query:",
            }
        ]

        r = requests.post(
f"{settings.OLLAMA_URL}/api/chat",
            json={
                "model": "mistral-7b-instruct",
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.3},
"keep_alive": settings.OLLAMA_KEEP_ALIVE
            },
            timeout=30
        )

        if r.status_code == 200:
            j = r.json()
            rewritten = (j.get("message") or {}).get("content", "").strip()
            if rewritten:
                for c in codes:
                    if c not in rewritten:
                        rewritten = (rewritten + " " + c).strip()
                orig_lower = prompt.lower()
                rew_lower = rewritten.lower()
                keywords = ["workweek", "work week", "pto", "decisions tool", "timekeeping"]
                for kw in keywords:
                    if kw in orig_lower and kw not in rew_lower:
                        rewritten = (rewritten + " " + kw).strip()
                        rew_lower = rewritten.lower()
                if not codes or all(c in rewritten for c in codes):
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
    # Optional filters (non-breaking)
    org: Optional[str] = None
    category: Optional[str] = None
    doc_code: Optional[str] = None
    owner: Optional[str] = None
    rewrite: bool = True

    # NEW: debug/display controls
    bypass_reranker: bool = False  # if true, skip cross-encoder step
    display_k: Optional[int] = None  # number of sources to show back to UI (formatting only)

    # Collection selection for /query
    collection: Optional[str] = Field(default="docs_v2", description="Chroma collection name to search")

class QueryResponse(BaseModel):
    """Response schema for the /query endpoint."""
    answer: str
    sources: List[Dict[str, Any]]
    original_query: str
    rewritten_query: str
    prompt_rewritten: bool
    alt_queries: List[str] = []
    hyde_prompt: Optional[str] = None
    retrieval_runs: List[Dict[str, Any]] = []
    fusion: List[Dict[str, Any]] = []
    rerank: List[Dict[str, Any]] = []
    final_context: List[Dict[str, Any]] = []
    phantom_citations_found: bool = False
    phantom_citation_details: List[Any] = []

    # NEW: UI-only formatting fields (do not affect retrieval)
    display_k: int = 0
    display_context: List[Dict[str, Any]] = []

def semantic_query(body: QueryBody) -> QueryResponse:
    """
    Semantic RAG flow:
      1) optional rewrite -> query variants
      2) dense retrieve per variant (K each)
      3) RRF fuse -> cutoff
      4) rerank (or bypass if requested)
      5) build grounded answer + validate/renumber citations
      6) return telemetry + UI-only display context
    """
    from config import settings  # local import to avoid circulars
    from adampy.pipeline.semantic_rag import (
        generate_query_variants,
        dense_retrieve,
        rrf_fuse,
        load_reranker_or_reuse,
        rerank,
        build_grounded_answer,
    )
    from adampy.pipeline.citations import validate_and_fix_citations

    # 1) Query variants (only if rewrite=True)
    ollama = OllamaClient()
    variants = (
        generate_query_variants(
            ollama,
            body.query,
            settings.SEMRAG_VARIANTS,
            settings.SEMRAG_USE_HYDE,
        )
        if body.rewrite
        else {"rewritten": None, "alternates": [], "hyde": None}
    )
    
    logger.debug("After ollama setup"); 
    logger.debug("Calling %s" , settings.OLLAMA_URL)

    query_set = [body.query]
    if variants.get("rewritten"):
        query_set.append(variants["rewritten"])
    query_set += variants.get("alternates", [])[: settings.SEMRAG_VARIANTS]
    if settings.SEMRAG_USE_HYDE and variants.get("hyde"):
        query_set.append(variants["hyde"])

    # Resolve collection for this request (default to settings or doc_v2)
    col_name = (body.collection or settings.COLLECTION or "docs_v2").strip()
    try:
        retr = _ensure_collection(col_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open collection '{col_name}': {e}")

    # 2) Dense retrieval per variant
    retrieved_runs = []
    per_query_results = []
    for q in query_set:
        try:
            hits = dense_retrieve(retr, q, settings.SEMRAG_K_PER_VARIANT)
        except Exception as e:
            msg = str(e)
            if "dimension" in msg.lower() or "expecting embedding" in msg.lower():
                # Reset the collection so future ingests/queries are consistent
                try:
                    client.delete_collection(col_name)
                except Exception:
                    pass
                _ = _ensure_collection(col_name)
                # Surface a clear message so the caller can trigger reingestion
                raise HTTPException(
                    status_code=409,
                    detail=f"Collection '{col_name}' had mismatched embedding dimension and was reset. Please re-ingest documents.",
                )
            raise
        retrieved_runs.append(
            {
                "query": q,
                "k": settings.SEMRAG_K_PER_VARIANT,
                "results": [
                    {
                        "doc_id": p.doc_id,
                        "chunk_id": p.chunk_id,
                        "score_dense": p.score_dense,
                        "rank": i + 1,
                        "title": p.title,
                        "url": p.url,
                        "section_heading": p.section_heading,
                        "page_num": p.page_num,
                    }
                    for i, p in enumerate(hits)
                ],
            }
        )
        per_query_results.append(hits)

    # 3) RRF fuse
    fused = rrf_fuse(per_query_results, settings.SEMRAG_RRF_CUTOFF)
    fusion_logs = [
        {
            "doc_id": p.doc_id,
            "chunk_id": p.chunk_id,
            "rrf_score": p.rrf_score,
            "fused_rank": i + 1,
        }
        for i, p in enumerate(fused)
    ]

    # 4) Rerank (or bypass)
    rerank_logs: List[Dict[str, Any]] = []
    if body.bypass_reranker:
        # Skip cross-encoder, just take top N by fused rank
        reranked = fused[: settings.SEMRAG_RERANK_KEEP]
    else:
        reranker = load_reranker_or_reuse()
        reranked = rerank(reranker, body.query, fused, settings.SEMRAG_RERANK_KEEP)
        rerank_logs = [
            {
                "doc_id": p.doc_id,
                "chunk_id": p.chunk_id,
                "rerank_score": p.rerank_score,
                "rerank_rank": i + 1,
            }
            for i, p in enumerate(reranked)
        ]

    logger.debug("About to call build_ground_answer")
    # 5) Build grounded answer + validate/repair citations
    answer_raw, final_context = build_grounded_answer(ollama, body.query, reranked)
    final_text, phantom_found, phantom_details = validate_and_fix_citations(
        answer_raw, reranked
    )

    # 6) UI display cap (formatting only)
    from config import settings as cfg
    display_k = body.display_k if (body.display_k is not None) else cfg.DISPLAY_TOP_K_DEFAULT
    display_context = final_context[: max(0, int(display_k))]

    return QueryResponse(
        answer=final_text,
        sources=final_context,  # keep full set for clients that still read this
        original_query=body.query,
        rewritten_query=variants.get("rewritten") or body.query,
        prompt_rewritten=bool(variants.get("rewritten")) and body.rewrite,
        alt_queries=variants.get("alternates", []),
        hyde_prompt=variants.get("hyde"),
        retrieval_runs=retrieved_runs,
        fusion=fusion_logs,
        rerank=rerank_logs,
        final_context=final_context,
        phantom_citations_found=phantom_found,
        phantom_citation_details=phantom_details,
        display_k=display_k,
        display_context=display_context,
    )

# Quick smoke tests:
# curl -s -X POST http://localhost:8000/query -H 'Content-Type: application/json' \
#   -d '{"query": "When does the standard workweek begin and end?"}'
# curl -s -X POST http://localhost:8000/query -H 'Content-Type: application/json' \
#   -d '{"query": "Per CLG-EN-PO-0301, where do I submit PTO?"}'
# curl -s -X POST http://localhost:8000/query -H 'Content-Type: application/json' \
#   -d '{"query": "What is the weather on Mars?"}'

@app.post("/query", response_model=QueryResponse)
def query_api(body: QueryBody) -> QueryResponse:
    if settings.SEMRAG_ENABLED:
        return semantic_query(body)
    start_total = _now_ms()
    code_pat = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+")
    logger.debug("In query")


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
   

    where: Optional[Dict[str, Any]] = None
    if body.org or body.category or body.doc_code or body.owner:
        where = {}
        if body.org:
            where["org"] = body.org
        if body.category:
            where["category"] = body.category
        if body.doc_code:
            where["doc_code"] = body.doc_code
        if body.owner:
            where["owner"] = body.owner
   

    # Resolve collection for this request (default to settings or doc_v2)
    col_name = (body.collection or settings.COLLECTION or "docs_v2").strip()
    try:
        retr = client.get_or_create_collection(name=col_name, embedding_function=CHROMA_EMBED)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open collection '{col_name}': {e}")

    rdebug: Dict[str, Any] = {}
    top_hits = hybrid_rerank(rewritten_query, retr, "bge-reranker-v2-m3", where=where, debug=rdebug)
    logger.debug("In Query top_hits: %s",  json.dumps(top_hits, indent=2))

    if not top_hits:
        final_answer = "I'm sorry, I couldn't find relevant information."
        final_answer, is_html_final, tags_final = _ensure_html(final_answer)
      
        return QueryResponse(
            answer=final_answer,
            sources=[],
            original_query=body.query,
            rewritten_query=rewritten_query,
            prompt_rewritten=prompt_rewritten,
        )
    hits: List[Dict[str, Any]] = []
    for h in top_hits:
        t = _get_text_for_hit(h)
        if t:
            h["text"] = t
            hits.append(h)
    if not hits:
        for h in top_hits:
            t = _get_text_for_hit(h)
            if t:
                h["text"] = t
                hits.append(h)
            if len(hits) >= 3:
                break
    logger.debug("In Query hits: %s",  json.dumps(hits, indent=2))
    if not hits:
        final_answer = (
            "<p>I couldn't load readable text from the retrieved sources. "
            "Please re-index the policy with a small text preview.</p>"
        )
      
        return QueryResponse(
            answer=final_answer,
            sources=[],
            original_query=body.query,
            rewritten_query=rewritten_query,
            prompt_rewritten=prompt_rewritten,
        )

    for i, h in enumerate(hits):
        logger.debug("hit: %s", json.dumps(h, indent=2))
        raw = h.get("text_content") or ""
        h["text"] = _extract_passage(raw, rewritten_query)
        h["index"] = i + 1


    start_ans = _now_ms()
    answer_first = ask_with_context(rewritten_query, hits, chat_history=body.history, model=body.model)
    ans_time = _now_ms() - start_ans
    answer_first, is_html_first, tags_first = _ensure_html(answer_first)
    used_first, valid_first = _extract_numeric_citations(answer_first, len(hits))
    citations_first = sorted(list(used_first))
   
    filtered = filter_cited_sources(answer_first, hits) if valid_first and used_first else []
    answer_used = answer_first
    is_html_final = is_html_first
    tags_final = tags_first
    if not valid_first or not filtered:
        extra = ""
        snippet_all = " ".join([h.get("text") or "" for h in hits])
        if "12:00" in snippet_all and "11:59" in snippet_all:
            extra += " Include both the start and end time and the total hours if mentioned."
        start_second = _now_ms()
        second = ask_with_context(
            rewritten_query,
            hits,
            chat_history=body.history,
            model=body.model,
            force_citations=True,
            extra_system_prompt=extra,
        )
        second_ms = _now_ms() - start_second
        second, is_html_second, tags_second = _ensure_html(second)
        used_second, valid_second = _extract_numeric_citations(second, len(hits))
        citations_second = sorted(list(used_second))
       
        if valid_second and citations_second:
            filtered = filter_cited_sources(second, hits)
            if filtered:
                answer_used = second
                is_html_final = is_html_second
                tags_final = tags_second
        if not filtered:
            final_answer = "I'm sorry, I can't answer confidently from the provided sources."
            final_answer, is_html_final, tags_final = _ensure_html(final_answer)
          
            return QueryResponse(
                answer=final_answer,
                sources=[],
                original_query=body.query,
                rewritten_query=rewritten_query,
                prompt_rewritten=prompt_rewritten,
            )
    else: logger.debug("Some thing bad happened")

    rich = []
    for h in filtered:
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
            "snippet": _get_text_for_hit(h)[:280],
        })

  
    return QueryResponse(
        answer=answer_used,
        sources=rich,
        original_query=body.query,
        rewritten_query=rewritten_query,
        prompt_rewritten=prompt_rewritten,
    )


@app.post("/upload")
def upload_api(file: UploadFile = File(...)):
    dest = _Path(settings.UPLOAD_DIR) / file.filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        f.write(file.file.read())
    n = upsert_document(dest, source="upload")
    return {"ok": True, "chunks": n, "path": str(dest)}


@app.post("/reindex")
def reindex_api():
    root = _Path(settings.WATCH_DIR)
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
    return {"ok": True, "collection": settings.COLLECTION, "count": cnt, "supported": list(SUPPORTED)}


## Removed: /delete endpoint


@app.post("/search")
def search_only(payload: dict = Body(...)):
    q = payload.get("query", "")
    k = int(payload.get("k", 4))
    return {"results": search(q, k=k)}


## Removed: /list endpoint


@app.post("/reset")
def reset_api():
    """Reset all Chroma collections, then recreate the default collection."""
    try:
        cols = []
        try:
            cols = client.list_collections()
        except Exception:
            cols = []
        for c in cols:
            try:
                # c can be a Collection object or dict-like depending on client version
                name = getattr(c, "name", None) or (c.get("name") if isinstance(c, dict) else None)
                if name:
                    client.delete_collection(name)
            except Exception:
                pass
    finally:
        # Recreate default collection for app health
        global collection
        collection = _ensure_collection()
    return {"ok": True, "reset": "all"}

@app.get("/ollama_health")
def ollama_health():
    try:
        r = requests.get(f"{settings.OLLAMA_URL}/api/tags", timeout=10)
        ct = r.headers.get("content-type", "")
        body = r.json() if "application/json" in ct else r.text
        return {"ok": r.status_code == 200, "status": r.status_code, "body": body}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama unreachable: {e}")

@app.get("/collections")
def list_collections():
    """List all Chroma collection names."""
    try:
        cols = []
        try:
            cols = client.list_collections()
        except Exception:
            cols = []
        names = []
        for c in cols:
            name = getattr(c, "name", None) or (c.get("name") if isinstance(c, dict) else None)
            if name:
                names.append(name)
        names = sorted(set(names))
        return {"count": len(names), "collections": names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {e}")

@app.get("/embed_health")
def embed_health():
    try:
        vec = EMBEDDER.encode(["hello world"])[0]
        return {"ok": True, "dim": len(vec), "preview": vec[:8]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embed error: {e}")


@app.get("/list_docs")
def list_documents(collection: Optional[str] = Query(None), limit: int = 10000):
    try:
        # Resolve collection name (default to configured one)
        col_name = (collection or settings.COLLECTION).strip()

        # Open the specified collection with the app's embedding function
        col = _ensure_collection()

        # Fetch all document entries with metadata
        results = col.get(include=["metadatas", "documents"], limit=limit)

        docs = []
        for i in range(len(results["ids"])):
            meta = results["metadatas"][i] or {}
            doc_info = {
                "doc_id": results["ids"][i],
                "metadata": meta,
            }
            docs.append(doc_info)

        return JSONResponse(content={"documents": docs, "count": len(docs), "collection": col_name})

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


if _Path(settings.WATCH_DIR).exists():
    obs = Observer()
    obs.schedule(Handler(), settings.WATCH_DIR, recursive=True)
    obs.start()


# ---------------- Ingest queue & jobs ----------------
INGEST_Q: "queue.Queue[tuple[str,str,str]]" = queue.Queue(maxsize=settings.INGEST_QUEUE_MAX)
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
    dest = _Path(settings.UPLOAD_DIR) / file.filename
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
## Removed downloads helper (unused after removing download endpoints)


## Removed: /download endpoint


## Removed: ZipRequest model (only used by removed download_zip)


## Removed: /download_zip endpoint



## Removed legacy /query_path; handler deleted
# Latest record (without snippets)
# curl -s http://localhost:8000/debug_last | jq .
#
# Latest with snippets (truncate to 400 chars)
# curl -s "http://localhost:8000/debug_last?include_snippets=true&max_chars=400" | jq .
#
# Third most recent
# curl -s "http://localhost:8000/debug_last?index=-3" | jq .
#
# With auth (if DEBUG_API_KEY=secret123 is set)
# curl -s -H "X-Debug-Key: secret123" "http://localhost:8000/debug_last?include_snippets=true" | jq .


@app.get("/debug_last")
def debug_last(index: int = -1,
               include_snippets: bool = False,
               max_chars: int = 280,
               x_debug_key: Optional[str] = Header(default=None)):
    if _DEBUG_API_KEY and x_debug_key != _DEBUG_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    count = len(_debug_buffer)
    if count == 0:
        return {"count": 0, "index": None, "record": None}
    if index < 0:
        idx = count + index
    else:
        idx = index
    if idx < 0 or idx >= count:
        raise HTTPException(status_code=400, detail="index out of range")
    record = _debug_buffer[idx]
    rec = json.loads(json.dumps(record))

    def sanitize(obj):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    out[k] = sanitize(v)
                elif isinstance(v, str):
                    if include_snippets:
                        out[k] = _truncate(v, max_chars)
                    else:
                        out[k] = "" if k in ("snippet", "answer", "raw_text", "answer_used") else v
                else:
                    out[k] = v
            return out
        elif isinstance(obj, list):
            return [sanitize(x) for x in obj]
        elif isinstance(obj, str):
            return _truncate(obj, max_chars) if include_snippets else obj
        else:
            return obj

    rec = sanitize(rec)
    return {"count": count, "index": idx, "record": rec}


@app.post("/debug_last/clear")
def debug_last_clear(x_debug_key: Optional[str] = Header(default=None)):
    if _DEBUG_API_KEY and x_debug_key != _DEBUG_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    with _debug_lock:
        cnt = len(_debug_buffer)
        _debug_buffer.clear()
    return {"cleared": cnt}


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

class IngestChunk(BaseModel):
    # SharePoint identity / metadata
    sp_web_url: Optional[str] = None
    sp_item_id: Optional[str] = None
    e_tag: Optional[str] = None

    title: Optional[str] = None
    org: Optional[str] = None
    org_code: Optional[str] = None
    category: Optional[str] = None
    doc_code: Optional[str] = None
    owner: Optional[str] = None
    version: Optional[str] = None

    revision_date: Optional[str] = None
    latest_review_date: Optional[str] = None
    document_review_date: Optional[str] = None
    review_approval_date: Optional[str] = None

    keywords: Optional[str] = None
    enterprise_keywords: List[str] = []
    association_ids: List[str] = []

    domain: Optional[str] = None
    allowed_groups: List[str] = []

    # Content identity
    file_name: Optional[str] = None

    # Content (Markdown text preferred; content_bytes as fallback)
    content_bytes: Optional[str] = None  # base64 if present
    text_content: Optional[str] = None   # markdown/plain text
    summary: Optional[str] = None

    # Chunking info coming from crawler
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    chunk_index: Optional[int] = None
    breadcrumbs: Optional[str] = None  # e.g., "Telecommuting Process"

    # Routing / storage hints
    persist: Optional[bool] = Field(default=True, description="If False, do not persist to disk")
    collection: Optional[str] = Field(default="docs_v2", description="Chroma collection name")

    # extra fields are allowed and will be carried into metadata
    class Config:
        extra = "allow"

class IngestRequest(BaseModel):
    # Accept either one chunk or many chunks in a single POST
    chunks: Optional[List[IngestChunk]] = None

    # Back-compat: allow a single object body (the old handler used this)
    # If provided, we will normalize it into a single chunk ingestion.
    sp_web_url: Optional[str] = None
    sp_item_id: Optional[str] = None
    e_tag: Optional[str] = None
    title: Optional[str] = None
    file_name: Optional[str] = None
    text_content: Optional[str] = None
    content_bytes: Optional[str] = None
    collection: Optional[str] = None
    chunk_index: Optional[int] = None
    breadcrumbs: Optional[str] = None

# ---------------- New: SharePoint-first ingest (no local persistence) ----------------
# ========================= Ingestion Endpoint (REPLACED) =========================
from fastapi import HTTPException
import base64
import time

def _ensure_text_from_payload(chunk: IngestChunk) -> str:
    """
    Prefer markdown/plain text from text_content.
    If missing, try to decode content_bytes (base64). If still missing, 400.
    """
    if chunk.text_content and chunk.text_content.strip():
        return chunk.text_content
    if chunk.content_bytes:
        try:
            return base64.b64decode(chunk.content_bytes).decode("utf-8", errors="ignore")
        except Exception:
            pass
    raise HTTPException(status_code=400, detail="No text_content or decodable content_bytes provided for ingestion.")

def _build_metadata(chunk: IngestChunk) -> Dict[str, Any]:
    """
    Flatten known fields + carry extras into metadata. This preserves SharePoint traceability.
    """
    md = {
        "sp_web_url": chunk.sp_web_url,
        "sp_item_id": chunk.sp_item_id,
        "e_tag": chunk.e_tag,
        "title": chunk.title,
        "org": chunk.org,
        "org_code": chunk.org_code,
        "category": chunk.category,
        "doc_code": chunk.doc_code,
        "owner": chunk.owner,
        "version": chunk.version,
        "revision_date": chunk.revision_date,
        "latest_review_date": chunk.latest_review_date,
        "document_review_date": chunk.document_review_date,
        "review_approval_date": chunk.review_approval_date,
        "keywords": chunk.keywords,
        "enterprise_keywords": chunk.enterprise_keywords,
        "association_ids": chunk.association_ids,
        "domain": chunk.domain,
        "allowed_groups": chunk.allowed_groups,
        "file_name": chunk.file_name,
        "summary": chunk.summary,
        "chunk_size": chunk.chunk_size,
        "chunk_overlap": chunk.chunk_overlap,
        "chunk_index": chunk.chunk_index,
        "chunk":  chunk.chunk_index, 
        "breadcrumbs": chunk.breadcrumbs,
        "ingested_at": int(time.time()),
        "text": chunk.text_content
    }
    # Include any extra fields passed by crawler
    for k, v in chunk.__dict__.items():
        if k not in md and not k.startswith("_"):
            md[k] = v
    # Ensure metadata conforms to Chroma's primitive-only type requirements
    return _sanitize_metadata(md)

def _make_doc_id(chunk: IngestChunk) -> str:
    base = chunk.sp_item_id or (chunk.file_name or f"anon-{int(time.time())}")
    idx = chunk.chunk_index if chunk.chunk_index is not None else 0
    return f"{base}:{idx}"

def _prepend_header_for_embedding(text: str, chunk: IngestChunk) -> str:
    """
    Give the retriever strong lexical/semantic anchors: DOC, PATH (breadcrumbs), FILE.
    """
    header = []
    if chunk.title:
        header.append(f"DOC: {chunk.title}")
    if chunk.breadcrumbs:
        header.append(f"PATH: {chunk.breadcrumbs}")
    if chunk.file_name:
        header.append(f"FILE: {chunk.file_name}")
    if header:
        return "\n".join(header) + "\n\n" + text
    return text

def _get_collection_name(chunk: IngestChunk) -> str:
    # default to settings.COLLECTION so ingest/query stay aligned
    return (chunk.collection or settings.COLLECTION).strip()

def _upsert_into_chroma(doc_id: str, text: str, metadata: Dict[str, Any], collection_name: str) -> None:
    """
    Insert/update a single chunk in Chroma. We reuse the existing embedding path if your app defines one;
    otherwise we give a clear error with instructions.
    """
    # Try to reuse an existing Chroma client / embedding function from your app
    try:
        # If you already created a global 'chroma_client' or 'retriever' elsewhere, reuse it.
        from adampy.pipeline.semantic_rag import get_chroma_client  # OPTIONAL helper you may already have
        client = get_chroma_client()
    except Exception:
        client = None

    if client is None:
        # Fallback: construct a local client using your existing retrieval client's storage
        try:
            import chromadb
            client = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_PATH", "./chroma"))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Chroma client unavailable: {e}")

    # Always ensure the collection uses our current embedding function
    try:
        col = _ensure_collection(collection_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open Chroma collection '{collection_name}': {e}")

    # Upsert
    try:
        col.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata],
        )
    except Exception as e:
        msg = str(e)
        # Auto-heal a dimension mismatch by resetting the collection, then ask for re-ingest
        if "dimension" in msg.lower() or "expecting embedding" in msg.lower():
            try:
                client.delete_collection(collection_name)
                _ = _ensure_collection(collection_name)
            except Exception:
                pass
            raise HTTPException(
                status_code=409,
                detail=f"Collection '{collection_name}' had mismatched embedding dimension. It was reset; please re-ingest.",
            )
        raise HTTPException(status_code=500, detail=f"Failed to upsert chunk {doc_id} into '{collection_name}': {e}")
class ResetCollectionBody(BaseModel):
    collection: Optional[str] = None


@app.post("/reset_collection")
def reset_collection(body: Optional[ResetCollectionBody] = None):
    """
    Delete and recreate a collection using the current embedding function.

    If no collection name is provided, resets settings.COLLECTION.
    """
    col_name = (body.collection if body else None) or settings.COLLECTION
    try:
        client.delete_collection(col_name)
    except Exception:
        pass

    client.get_or_create_collection(col_name, embedding_function=CHROMA_EMBED)
    return {"status": "ok", "collection": col_name}

@app.post("/ingest_document")
def ingest_document(req: IngestRequest):
    """
    Accepts either:
      - {'chunks': [IngestChunk, ...]}  # new crawler (chunked, markdown)
      - single-object body (backward compatibility)

    Each chunk is upserted into the specified Chroma collection (default: docs_v2).
    """

    logger.debug("Received IngestRequest: %s", json.dumps(req.dict(), indent=2))

    # Normalize request into a list of chunks
    if req.chunks and len(req.chunks) > 0:
        chunks = req.chunks
    else:
        # Back-compat: single object body → one chunk
        single = IngestChunk(
            sp_web_url=req.sp_web_url,
            sp_item_id=req.sp_item_id,
            e_tag=req.e_tag,
            title=req.title,
            file_name=req.file_name,
            text_content=req.text_content,
            content_bytes=req.content_bytes,
            collection=req.collection or settings.COLLECTION,
            chunk_index=req.chunk_index or 0,
            breadcrumbs=req.breadcrumbs,
        )
        chunks = [single]

    ingested = []
    for ch in chunks:
        # Ensure we have content
        logger.debug("_ensure_text_from_payload")
        text = _ensure_text_from_payload(ch)
        logger.debug("_prepend_header_for_embedding")
        text = _prepend_header_for_embedding(text, ch)
        logger.debug("_build_metadata")
        # Prepare metadata and id
        metadata = _build_metadata(ch)
        logger.debug("_make_doc_id")
        doc_id = _make_doc_id(ch)
        logger.debug("_get_collection_name(ch)")
        collection_name = _get_collection_name(ch)
        logger.debug("collection_name:" + collection_name)
        # Upsert into Chroma
        logger.debug("_upsert_into_chroma(doc_id, text, metadata, collection_name)")
        _upsert_into_chroma(doc_id, text, metadata, collection_name)
        logger.debug("_upsert_into_chroma completed")
        ingested.append({
            "id": doc_id,
            "collection": collection_name,
            "title": ch.title,
            "file_name": ch.file_name,
            "chunk_index": ch.chunk_index or 0,
            "sp_item_id": ch.sp_item_id,
            "url": ch.sp_web_url,
        })

    return {"success": True,  "chunks": len(ingested)}


@app.post("/infer_metadata", response_model=InferMetadataResponse)
def infer_metadata(req: InferMetadataRequest):
    # 1) Basic cleanup & length guard (reuse your cleaner)
    raw = (req.text or "").strip()
    cleaned = clean_document_text(raw)  # you already have this helper
    if len(cleaned) < 300:
        raise HTTPException(status_code=400, detail="Document text too short (<300 chars) after cleanup.")

    # 2) Ask your existing LLM summarizer (already wired to SUMMARY_MODEL)
    summary, category, keywords = summarize_document(cleaned)

    # 3) Normalize category to a controlled vocabulary
    vocab = set(_DEFAULT_CATEGORY_VOCAB)
    if req.category_vocab:
        vocab |= {v.strip().lower() for v in req.category_vocab if v and v.strip()}
    category_norm = _normalize_category(category, vocab)

    # 4) Compose initial keywords (seed with doc_code/title if present)
    seed_kw = list(keywords or [])
    if req.doc_code and req.doc_code not in seed_kw:
        seed_kw.append(req.doc_code)
    if req.title and req.title not in seed_kw:
        seed_kw.append(req.title)

    # 5) Apply anchor rules (Decisions/Time Off Request → PTO; J375 → personal leave)
    kw_rule_applied, hits = _apply_anchor_rules(cleaned, seed_kw, vocab, req.apply_rules)

    # 6) Final clean, cap, and return
    final_kw = _clean_keywords(
        kw_rule_applied,
        max_n=max(3, min(req.max_keywords, 32)),
        lower=req.lower_keywords,
        dedupe=req.dedupe_keywords,
    )

    # Make sure we always carry the normalized category as a keyword anchor
    if category_norm not in {k.lower() for k in final_kw}:
        final_kw = [category_norm] + final_kw
        # keep under limit
        final_kw = final_kw[:max(3, min(req.max_keywords, 32))]

    return InferMetadataResponse(
        summary=(summary or "").strip(),
        category=category_norm,
        keywords=final_kw,
        debug={"rule_hits": hits}
    )
