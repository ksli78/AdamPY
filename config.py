# config.py — no environment dependency, simple frozen settings

from dataclasses import dataclass, replace, field
from typing import Dict, List, Optional

@dataclass(frozen=True)
class Settings:
    # ---- Retrieval / RAG knobs ----
    SEMRAG_ENABLED: bool = True
    SEMRAG_VARIANTS: int = 3
    SEMRAG_K_PER_VARIANT: int = 50
    SEMRAG_RRF_CUTOFF: int = 200
    SEMRAG_RERANK_KEEP: int = 10
    SEMRAG_USE_HYDE: bool = False

    # ---- Ollama + generation ----
    OLLAMA_URL: str = "http://127.0.0.1:11434"
    OLLAMA_KEEP_ALIVE: str = "720m"

    CHAT_MODEL: str = "llama3:8b"
    SUMMARY_MODEL: str = "mistral-7b-instruct"

    # Optional: map friendly names to real models (kept purely in file)
    ALIAS_MAP: Dict[str, str] = field(default_factory=lambda: {
        "Adam Lite": "adam-lite:latest",
        "adam-lite": "adam-lite:latest",
        "llama3:8b": "llama3:8b",
        "mistral-7b-instruct": "mistral-7b-instruct:latest",
    })

    # Sampling and limits
    OLLAMA_TEMPERATURE: float = 0.1
    OLLAMA_TOP_P: float = 0.9
    OLLAMA_TOP_K: int = 40
    OLLAMA_REPEAT_PENALTY: float = 1.1
    OLLAMA_NUM_PREDICT: int = 1280
    OLLAMA_SEED: Optional[int] = None
    OLLAMA_PRESENCE_PENALTY: float = 0.0
    OLLAMA_FREQUENCY_PENALTY: float = 0.0
    OLLAMA_MIROSTAT: int = 0              # 0=off, 1 or 2 to enable
    OLLAMA_MIROSTAT_TAU: float = 5.0
    OLLAMA_MIROSTAT_ETA: float = 0.1
    OLLAMA_STOP: Optional[List[str]] = None  # e.g., ["###", "</s>"]

    # ---- Reranker / embeddings / storage ----
    RERANKER_MODEL_PATH: str = "/opt/rag-models/bge-reranker-v2-m3"
    RERANKER_BATCH_SIZE: int = 16
    RERANKER_MAX_LEN: int = 512

    EMBED_MODEL_DIR: str = "/opt/adam/models/nomic-ai/nomic-embed-text"
    CHROMA_DIR: str = "/srv/rag/chroma"
    COLLECTION: str = "docs_v2"

    # ---- Filesystem / ingest ----
    WATCH_DIR: str = "/srv/rag/watched"
    UPLOAD_DIR: str = "/srv/rag/uploads"
    INGEST_QUEUE_MAX: int = 8

    # ---- Chunking ----
    CHUNK_SIZE: int = 350
    CHUNK_OVERLAP: int = 150

    # ---- UI / validation ----
    DISPLAY_TOP_K_DEFAULT: int = 10
    STRICT_CITATION_CHECKS: bool = True

    # Back-compat shim: property alias for legacy usages
    @property
    def OLLAMA_HOST(self) -> str:  # pragma: no cover - simple alias
        return self.OLLAMA_URL


# Base settings object (immutable)
_base = Settings()

# Optional local overrides without using env vars.
# Create a sibling file config_local.py with:
#   OVERRIDES = {"CHAT_MODEL": "llama3:8b", "OLLAMA_URL": "http://127.0.0.1:11434"}
try:
    from config_local import OVERRIDES  # type: ignore
except Exception:
    OVERRIDES = {}

def _apply_overrides(base: Settings, overrides: dict) -> Settings:
    # Only accept keys that exist on Settings; ignore unknown keys.
    valid = {k: v for k, v in overrides.items() if hasattr(base, k)}
    return replace(base, **valid) if valid else base

settings: Settings = _apply_overrides(_base, OVERRIDES)

# Safety guard: never allow the cursed "company-default" to slip in here.
if settings.CHAT_MODEL.strip().lower() == "company-default":
    raise RuntimeError(
        "Invalid CHAT_MODEL 'company-default' in config.py/config_local.py; "
        "tag it in Ollama or use a real model id."
    )
