#!/usr/bin/env bash
# set_rag_env.sh — reset then (re)export clean RAG env

set -euo pipefail

############################################
# 1) Hard reset: unset all related vars
############################################
# Unset by explicit names (common culprits)
unset CHAT_MODEL || true
unset SUMMARY_MODEL || true
unset COLLECTION || true

# Bulk-unset by prefix patterns
# shellcheck disable=SC2046
for VAR in $(env | cut -d= -f1 | egrep '^(OLLAMA_.*|SEMRAG_.*|RERANKER_.*|EMBED_.*|CHROMA_DIR|WATCH_DIR|UPLOAD_DIR|INGEST_QUEUE_MAX|CHUNK_.*|STRICT_CITATION_CHECKS|DISPLAY_TOP_K_DEFAULT)$'); do
  unset "$VAR" || true
done

############################################
# 2) Fresh exports (authoritative values)
############################################
set -a  # auto-export everything below

# Retrieval knobs
SEMRAG_ENABLED=true
SEMRAG_VARIANTS=3
SEMRAG_K_PER_VARIANT=50
SEMRAG_RRF_CUTOFF=200
SEMRAG_RERANK_KEEP=10
SEMRAG_USE_HYDE=false

# Ollama + reranker
# NOTE: If you run Ollama elsewhere, update OLLAMA_URL here.
OLLAMA_URL="http://127.0.0.1:11434"
OLLAMA_KEEP_ALIVE="720m"

RERANKER_MODEL_PATH="/opt/rag-models/bge-reranker-v2-m3"
RERANKER_BATCH_SIZE=16
RERANKER_MAX_LEN=512

# Embeddings + storage
EMBED_MODEL_DIR="/opt/adam/models/nomic-ai/nomic-embed-text"
CHROMA_DIR="/srv/rag/chroma"
COLLECTION="docs_v2"

# Filesystem
WATCH_DIR="/srv/rag/watched"
UPLOAD_DIR="/srv/rag/uploads"
INGEST_QUEUE_MAX=8

# Chunking
CHUNK_SIZE=350
CHUNK_OVERLAP=150

# Models (authoritative; do NOT leave aliases or “company-default” here)
CHAT_MODEL="llama3:8b"
SUMMARY_MODEL="mistral-7b-instruct"

# Generation options
STRICT_CITATION_CHECKS=true
OLLAMA_TEMPERATURE=0.1
OLLAMA_TOP_P=0.9
OLLAMA_TOP_K=40
OLLAMA_REPEAT_PENALTY=1.1
OLLAMA_NUM_PREDICT=1280
# Optional stops (comma-separated or leave empty)
# OLLAMA_STOP="###,</s>"

# UI
DISPLAY_TOP_K_DEFAULT=10

set +a  # stop auto-export

############################################
# 3) Sanity print (helps catch “company-default” quickly)
############################################
echo "[set_rag_env] OLLAMA_URL=${OLLAMA_URL}"
echo "[set_rag_env] CHAT_MODEL=${CHAT_MODEL}"
echo "[set_rag_env] SUMMARY_MODEL=${SUMMARY_MODEL}"
echo "[set_rag_env] COLLECTION=${COLLECTION}"
echo "[set_rag_env] Environment refreshed."
