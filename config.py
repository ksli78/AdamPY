import os
from typing import Optional

import chromadb

from embedding import NomicOnnxEmbedder

# Environment configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "45m")
EMBED_MODEL_DIR = os.getenv("EMBED_MODEL_DIR", "/opt/adam/models/nomic-ai/nomic-embed-text")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3:8b")
CHROMA_DIR = os.getenv("CHROMA_DIR", "/srv/rag/chroma")
WATCH_DIR = os.getenv("WATCH_DIR", "/srv/rag/watched")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/srv/rag/uploads")
COLLECTION = os.getenv("COLLECTION", "company_docs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))
Q_MAX = int(os.getenv("INGEST_QUEUE_MAX", "8"))

# Model aliasing
ALIAS_MAP = {
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


# Initialize embedder and vector collection
EMBEDDER = NomicOnnxEmbedder(EMBED_MODEL_DIR)
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(COLLECTION)

# Supported file extensions
SUPPORTED = (
    ".pdf", ".docx", ".txt",
    ".md", ".csv", ".log", ".json", ".py", ".cs", ".java",
    ".html", ".htm",
    ".eml", ".msg",
    ".xlsx", ".pptx",
    ".png", ".jpg", ".jpeg", ".tiff", ".tif",
    ".doc", ".rtf",
)
