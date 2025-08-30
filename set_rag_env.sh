#!/bin/bash
# Environment overrides for Semantic RAG

# Retrieval knobs
export SEMRAG_ENABLED=true
export SEMRAG_VARIANTS=3
export SEMRAG_K_PER_VARIANT=50
export SEMRAG_RRF_CUTOFF=200
export SEMRAG_RERANK_KEEP=10
export SEMRAG_USE_HYDE=false

# Ollama + reranker
export OLLAMA_URL="http://127.0.0.1:11434"
export OLLAMA_KEEP_ALIVE="720m"
export RERANKER_MODEL_PATH="/opt/rag-models/bge-reranker-v2-m3"
export RERANKER_BATCH_SIZE=16
export RERANKER_MAX_LEN=512

# Embeddings + storage
export EMBED_MODEL_DIR="/opt/adam/models/nomic-ai/nomic-embed-text"
export CHROMA_DIR="/srv/rag/chroma"
export COLLECTION="docs_v2"     # ?? make sure this matches what you ingested into

# Filesystem
export WATCH_DIR="/srv/rag/watched"
export UPLOAD_DIR="/srv/rag/uploads"
export INGEST_QUEUE_MAX=8

# Chunking (only matters on re-ingest, but can be set here for consistency)
export CHUNK_SIZE=350
export CHUNK_OVERLAP=150

# Models
export CHAT_MODEL="llama3:8b"
export SUMMARY_MODEL="mistral-7b-instruct"

# UI
export DISPLAY_TOP_K_DEFAULT=10

# Strict citations
export STRICT_CITATION_CHECKS=true
export OLLAMA_TEMPERATURE=0.1
export OLLAMA_TOP_P=0.9
export OLLAMA_TOP_K=40
export OLLAMA_REPEAT_PENALTY=1.1
export OLLAMA_NUM_PREDICT=1280
echo "RAG environment variables set."
