from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    SEMRAG_ENABLED: bool = True
    SEMRAG_VARIANTS: int = 2
    SEMRAG_K_PER_VARIANT: int = 10
    SEMRAG_RRF_CUTOFF: int = 50
    SEMRAG_RERANK_KEEP: int = 10
    SEMRAG_USE_HYDE: bool = False
    OLLAMA_HOST: str = "http://127.0.0.1:11434"
    RERANKER_MODEL_PATH: str = "/opt/models/bge-reranker-base"
    RERANKER_BATCH_SIZE: int = 16
    RERANKER_MAX_LEN: int = 512
    STRICT_CITATION_CHECKS: bool = True

    class Config:
        env_prefix = ""

settings = Settings()
