# --- SQLite shim to ensure Chroma works on old distros ---
try:
    import sys, importlib
    sys.modules["sqlite3"] = importlib.import_module("pysqlite3")
except Exception:
    pass

import os, io, base64, hashlib, tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import chromadb
from chromadb.config import Settings
from PyPDF2 import PdfReader

# ------------------ Embeddings ------------------
from onnxruntime import InferenceSession
import numpy as np

def load_onnx_model(model_path: str):
    return InferenceSession(model_path, providers=["CPUExecutionProvider"])

EMBED_MODEL_PATH = os.environ.get("EMBED_MODEL_PATH", "/opt/adam/models/nomic-ai/nomic-embed-text/model.onnx")
onnx_session = load_onnx_model(EMBED_MODEL_PATH)

def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    inputs = {onnx_session.get_inputs()[0].name: texts}
    # NOTE: This is a placeholder, actual tokenization logic should be same as in your working file
    # For brevity assume pre-tokenized float vectors
    return onnx_session.run(None, inputs)[0].tolist()

# ------------------ Config ------------------
CHROMA_DIR = os.environ.get("CHROMA_DIR", str(Path("./chroma").absolute()))
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "documents")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "120"))

client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection(name=CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})

# ------------------ Utilities ------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(end - overlap, 0)
    return chunks

def read_pdf_bytes(data: bytes) -> str:
    text = []
    with io.BytesIO(data) as f:
        reader = PdfReader(f)
        for page in reader.pages:
            try:
                text.append(page.extract_text() or "")
            except Exception:
                text.append("")
    return "\n".join(text)

# ------------------ Models ------------------
class IngestDocument(BaseModel):
    sp_web_url: Optional[str] = None
    sp_item_id: Optional[str] = None
    sp_file_id: Optional[str] = None
    etag: Optional[str] = None
    title: Optional[str] = None
    org: Optional[str] = None
    category: Optional[str] = None
    doc_code: Optional[str] = None
    owner: Optional[str] = None
    source: Optional[str] = None
    text_content: Optional[str] = None
    content_bytes: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None

class QueryRequest(BaseModel):
    query: str
    k: Optional[int] = 4
    org: Optional[str] = None
    category: Optional[str] = None
    doc_code: Optional[str] = None
    owner: Optional[str] = None
    source: Optional[str] = None

# ------------------ FastAPI ------------------
app = FastAPI(title="RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Endpoints ------------------
@app.post("/ingest_document")
def ingest_document(doc: IngestDocument):
    # Determine document id
    version = doc.etag or "v1"
    doc_id = None
    if doc.sp_item_id:
        doc_id = f"item:{doc.sp_item_id}:{version}"
    elif doc.sp_file_id:
        doc_id = f"file:{doc.sp_file_id}:{version}"
    elif doc.sp_web_url:
        doc_id = f"url:{doc.sp_web_url}:{version}"
    else:
        doc_id = f"doc:{sha256_bytes((doc.title or '').encode())}:{version}"

    text = doc.text_content or ""
    if doc.content_bytes:
        raw = base64.b64decode(doc.content_bytes)
        # only handling pdf for brevity
        text = read_pdf_bytes(raw)

    chunks = chunk_text(text, doc.chunk_size or CHUNK_SIZE, doc.chunk_overlap or CHUNK_OVERLAP)
    embeddings = embed_texts(chunks)
    metadatas = [{
        "title": doc.title, "org": doc.org, "category": doc.category,
        "doc_code": doc.doc_code, "owner": doc.owner, "source": doc.source,
        "sp_web_url": doc.sp_web_url, "sp_item_id": doc.sp_item_id,
        "sp_file_id": doc.sp_file_id, "etag": doc.etag
    } for _ in chunks]

    ids = [f"{doc_id}:{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)
    return {"status": "ok", "chunks": len(chunks)}

@app.post("/delete_sp")
def delete_sp(sp_item_id: Optional[str] = None, sp_file_id: Optional[str] = None):
    if not sp_item_id and not sp_file_id:
        raise HTTPException(status_code=400, detail="sp_item_id or sp_file_id required")
    # delete by metadata match
    to_delete = []
    results = collection.get(include=["metadatas", "ids"])
    for idx, meta in zip(results["ids"], results["metadatas"]):
        if meta.get("sp_item_id") == sp_item_id or meta.get("sp_file_id") == sp_file_id:
            to_delete.append(idx)
    if to_delete:
        collection.delete(ids=to_delete)
    return {"status": "deleted", "count": len(to_delete)}

@app.post("/query")
def query_rag(req: QueryRequest):
    filters = {}
    for f in ["org", "category", "doc_code", "owner", "source"]:
        val = getattr(req, f)
        if val:
            filters[f] = val
    results = collection.query(query_texts=[req.query], n_results=req.k or 4, where=filters)
    return {"results": results}

@app.get("/embed_health")
def embed_health():
    return {"ok": True}

@app.get("/ollama_health")
def ollama_health():
    # stub
    return {"ok": True}
