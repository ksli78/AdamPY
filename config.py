import os
from dataclasses import dataclass

try:
    from pydantic_settings import BaseSettings  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    BaseSettings = None  # type: ignore

if BaseSettings:
    class Settings(BaseSettings):
        SEMRAG_ENABLED: bool = True
        SEMRAG_VARIANTS: int = 2
        SEMRAG_K_PER_VARIANT: int = 20
        SEMRAG_RRF_CUTOFF: int = 100
        SEMRAG_RERANK_KEEP: int = 15
        SEMRAG_USE_HYDE: bool = False
        COLLECTION_PRIMARY: str = "docs_v2"
        COLLECTION_FALLBACK: str = "docs_v1"
        USE_COLLECTION_FALLBACK: bool = True
        OLLAMA_HOST: str = "http://127.0.0.1:11434"
        RERANKER_MODEL_PATH: str = "/opt/rag-models/bge-reranker-v2-m3"
        RERANKER_BATCH_SIZE: int = 16
        RERANKER_MAX_LEN: int = 512
        STRICT_CITATION_CHECKS: bool = True
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
        SEMRAG_ENABLED: bool = _get_bool("SEMRAG_ENABLED", True)
        SEMRAG_VARIANTS: int = _get_int("SEMRAG_VARIANTS", 2)
        SEMRAG_K_PER_VARIANT: int = _get_int("SEMRAG_K_PER_VARIANT", 20)
        SEMRAG_RRF_CUTOFF: int = _get_int("SEMRAG_RRF_CUTOFF", 100)
        SEMRAG_RERANK_KEEP: int = _get_int("SEMRAG_RERANK_KEEP", 15)
        SEMRAG_USE_HYDE: bool = _get_bool("SEMRAG_USE_HYDE", False)
        COLLECTION_PRIMARY: str = os.getenv("COLLECTION_PRIMARY", "docs_v2")
        COLLECTION_FALLBACK: str = os.getenv("COLLECTION_FALLBACK", "docs_v1")
        USE_COLLECTION_FALLBACK: bool = _get_bool("USE_COLLECTION_FALLBACK", True)
        OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        RERANKER_MODEL_PATH: str = os.getenv("RERANKER_MODEL_PATH", "/opt/rag-models/bge-reranker-v2-m3")
        RERANKER_BATCH_SIZE: int = _get_int("RERANKER_BATCH_SIZE", 16)
        RERANKER_MAX_LEN: int = _get_int("RERANKER_MAX_LEN", 512)
        STRICT_CITATION_CHECKS: bool = _get_bool("STRICT_CITATION_CHECKS", True)
        DISPLAY_TOP_K_DEFAULT: int = _get_int("DISPLAY_TOP_K_DEFAULT", 5)

    settings = Settings()
