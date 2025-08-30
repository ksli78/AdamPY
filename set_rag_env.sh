#!/usr/bin/env bash
# set_rag_env.sh — safe to source; resets only our app vars, then exports fresh ones.
# IMPORTANT: no `set -e`/`pipefail` here — we don't want to kill the parent shell.

# --- 1) Unset our app-related variables (only those we own) ---
_safe_unset() { unset "$1" 2>/dev/null || true; }

# Explicit common names
for v in CHAT_MODEL SUMMARY_MODEL COLLECTION; do _safe_unset "$v"; done

# Patterns (exported vars only)
while IFS='=' read -r name _; do
  case "$name" in
    OLLAMA_*|SEMRAG_*|RERANKER_*|EMBED_*|CHROMA_DIR|WATCH_DIR|UPLOAD_DIR|INGEST_QUEUE_MAX|CHUNK_*|STRICT_CITATION_CHECKS|DISPLAY_TOP_K_DEFAULT)
      _safe_unset "$name"
      ;;
  esac
done < <(env)

# --- 2) Export fresh values (authoritative) ---
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

# ✔ Set concrete model IDs; no aliases, no "company-default"
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
