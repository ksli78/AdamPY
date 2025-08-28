import os
from dataclasses import dataclass

try:
    from pydantic_settings import BaseSettings  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    BaseSettings = None  # type: ignore

if BaseSettings:
    class Settings(BaseSettings):
        # Semantic RAG controls
        SEMRAG_ENABLED: bool = True
        SEMRAG_VARIANTS: int = 2
        SEMRAG_K_PER_VARIANT: int = 20
        SEMRAG_RRF_CUTOFF: int = 100
        SEMRAG_RERANK_KEEP: int = 15
        SEMRAG_USE_HYDE: bool = False

        # Ollama + reranker
        OLLAMA_URL: str = "http://127.0.0.1:11434"
        OLLAMA_KEEP_ALIVE: str = "720m"
        RERANKER_MODEL_PATH: str = "/opt/rag-models/bge-reranker-v2-m3"
        RERANKER_BATCH_SIZE: int = 16
        RERANKER_MAX_LEN: int = 512

        # Embeddings + storage
        EMBED_MODEL_DIR: str = "/opt/adam/models/nomic-ai/nomic-embed-text"
        CHROMA_DIR: str = "/srv/rag/chroma"
        COLLECTION: str = "company_docs"

        # Filesystem
        WATCH_DIR: str = "/srv/rag/watched"
        UPLOAD_DIR: str = "/srv/rag/uploads"
        INGEST_QUEUE_MAX: int = 8

        # Chunking
        CHUNK_SIZE: int = 1400
        CHUNK_OVERLAP: int = 300

        # Models
        CHAT_MODEL: str = "llama3:8b"
        SUMMARY_MODEL: str = "mistral-7b-instruct"

        # Model aliasing
        ALIAS_MAP: dict = {
            "Adam Lite": "adam-lite:latest",
            "adam-lite": "adam-lite:latest",
            "llama3:8b": "llama3:8b",
            "mistral-7b-instruct": "mistral-7b-instruct:latest",
        }

        STRICT_CITATION_CHECKS: bool = True

        # UI display cap (formatting only)
        DISPLAY_TOP_K_DEFAULT: int = 5

        class Config:
            env_prefix = ""

    settings = Settings()
else:
    def _get_bool(name: str, default: bool) -> bool:
        return os.getenv(name, str(int(default))).lower() in {"1", "true", "yes"}

    def _get_int(name: str, default: int) -> int:
        return int(os.getenv(name, str(default)))

    @dataclass
    class Settings:
        # Semantic RAG controls
        SEMRAG_ENABLED: bool = _get_bool("SEMRAG_ENABLED", True)
        SEMRAG_VARIANTS: int = _get_int("SEMRAG_VARIANTS", 2)
        SEMRAG_K_PER_VARIANT: int = _get_int("SEMRAG_K_PER_VARIANT", 20)
        SEMRAG_RRF_CUTOFF: int = _get_int("SEMRAG_RRF_CUTOFF", 100)
        SEMRAG_RERANK_KEEP: int = _get_int("SEMRAG_RERANK_KEEP", 15)
        SEMRAG_USE_HYDE: bool = _get_bool("SEMRAG_USE_HYDE", False)

        # Ollama + reranker
        OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
        OLLAMA_KEEP_ALIVE: str = os.getenv("OLLAMA_KEEP_ALIVE", "720m")
        RERANKER_MODEL_PATH: str = os.getenv("RERANKER_MODEL_PATH", "/opt/rag-models/bge-reranker-v2-m3")
        RERANKER_BATCH_SIZE: int = _get_int("RERANKER_BATCH_SIZE", 16)
        RERANKER_MAX_LEN: int = _get_int("RERANKER_MAX_LEN", 512)

        # Embeddings + storage
        EMBED_MODEL_DIR: str = os.getenv("EMBED_MODEL_DIR", "/opt/adam/models/nomic-ai/nomic-embed-text")
        CHROMA_DIR: str = os.getenv("CHROMA_DIR", "/srv/rag/chroma")
        COLLECTION: str = os.getenv("COLLECTION", "company_docs")

        # Filesystem
        WATCH_DIR: str = os.getenv("WATCH_DIR", "/srv/rag/watched")
        UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "/srv/rag/uploads")
        INGEST_QUEUE_MAX: int = _get_int("INGEST_QUEUE_MAX", 8)

        # Chunking
        CHUNK_SIZE: int = _get_int("CHUNK_SIZE", 1400)
        CHUNK_OVERLAP: int = _get_int("CHUNK_OVERLAP", 300)

        # Models
        CHAT_MODEL: str = os.getenv("CHAT_MODEL", "llama3:8b")
        SUMMARY_MODEL: str = os.getenv("SUMMARY_MODEL", "mistral-7b-instruct")

        # Model aliasing
        ALIAS_MAP: dict = None  # filled in __post_init__

        STRICT_CITATION_CHECKS: bool = _get_bool("STRICT_CITATION_CHECKS", True)

        # UI display cap (formatting only)
        DISPLAY_TOP_K_DEFAULT: int = _get_int("DISPLAY_TOP_K_DEFAULT", 5)

        def __post_init__(self):
            self.ALIAS_MAP = {
                "Adam Lite": "adam-lite:latest",
                "adam-lite": "adam-lite:latest",
                "llama3:8b": "llama3:8b",
                "mistral-7b-instruct": "mistral-7b-instruct:latest",
            }

    settings = Settings()
