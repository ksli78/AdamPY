# set_rag_env_minimal.sh  (POSIX-safe; source this)
# 1) Unset known app vars (explicit list only)
unset CHAT_MODEL SUMMARY_MODEL COLLECTION
unset OLLAMA_URL OLLAMA_KEEP_ALIVE
unset RERANKER_MODEL_PATH RERANKER_BATCH_SIZE RERANKER_MAX_LEN
unset EMBED_MODEL_DIR CHROMA_DIR WATCH_DIR UPLOAD_DIR INGEST_QUEUE_MAX
unset CHUNK_SIZE CHUNK_OVERLAP
unset SEMRAG_ENABLED SEMRAG_VARIANTS SEMRAG_K_PER_VARIANT SEMRAG_RRF_CUTOFF SEMRAG_RERANK_KEEP SEMRAG_USE_HYDE
unset DISPLAY_TOP_K_DEFAULT STRICT_CITATION_CHECKS
unset OLLAMA_TEMPERATURE OLLAMA_TOP_P OLLAMA_TOP_K OLLAMA_REPEAT_PENALTY OLLAMA_NUM_PREDICT
# add any other noisy vars here if needed; keep it explicit and boring

# 2) Fresh exports (authoritative)
export SEMRAG_ENABLED=true
export SEMRAG_VARIANTS=3
export SEMRAG_K_PER_VARIANT=50
export SEMRAG_RRF_CUTOFF=200
export SEMRAG_RERANK_KEEP=10
export SEMRAG_USE_HYDE=false

export OLLAMA_URL="http://127.0.0.1:11434"
export OLLAMA_KEEP_ALIVE="720m"

export RERANKER_MODEL_PATH="/opt/rag-models/bge-reranker-v2-m3"
export RERANKER_BATCH_SIZE=16
export RERANKER_MAX_LEN=512

export EMBED_MODEL_DIR="/opt/adam/models/nomic-ai/nomic-embed-text"
export CHROMA_DIR="/srv/rag/chroma"
export COLLECTION="docs_v2"

export WATCH_DIR="/srv/rag/watched"
export UPLOAD_DIR="/srv/rag/uploads"
export INGEST_QUEUE_MAX=8

export CHUNK_SIZE=350
export CHUNK_OVERLAP=150

# concrete model ids; no aliases; no "company-default"
export CHAT_MODEL="llama3:8b"
export SUMMARY_MODEL="mistral-7b-instruct"

export STRICT_CITATION_CHECKS=true
export OLLAMA_TEMPERATURE=0.1
export OLLAMA_TOP_P=0.9
export OLLAMA_TOP_K=40
export OLLAMA_REPEAT_PENALTY=1.1
export OLLAMA_NUM_PREDICT=1280

export DISPLAY_TOP_K_DEFAULT=10

echo "[set_rag_env] refreshed. CHAT_MODEL=$CHAT_MODEL  OLLAMA_URL=$OLLAMA_URL"
