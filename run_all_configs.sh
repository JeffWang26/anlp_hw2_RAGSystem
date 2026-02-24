#!/bin/bash
# Full experiment commands (use GPU).
# Run from project root: bash run_all_configs.sh

set -e
ANDREW_ID="Porygon"
Q="leaderboard_queries.json"
OUT="system_outputs"

# ========== Baseline: MiniLM + hybrid + generative (Mistral) ==========
python run_rag.py --use-gpu --output $OUT/baseline_minilm_hybrid_generative.json \
  --retrieval hybrid --reader generative \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --andrew-id $ANDREW_ID --queries $Q

# ========== 1. extractive reader ==========
python run_rag.py --use-gpu --output $OUT/1_extractive_minilm_hybrid.json \
  --retrieval hybrid --reader extractive \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --andrew-id $ANDREW_ID --queries $Q

# ========== 2. embedding model (BGE) ==========
python run_rag.py --use-gpu --output $OUT/2_bge_hybrid_generative.json \
  --retrieval hybrid --reader generative \
  --embedding-model BAAI/bge-base-en-v1.5 \
  --andrew-id $ANDREW_ID --queries $Q

# ========== 3. generative model (Qwen2) ==========
python run_rag.py --use-gpu --output $OUT/3_minilm_hybrid_qwen2.json \
  --retrieval hybrid --reader generative \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --generative-model Qwen/Qwen2-7B-Instruct \
  --andrew-id $ANDREW_ID --queries $Q

# ========== 4. Dense vs Sparse vs Hybrid (all + generative) ==========
# 4a. Dense-only
python run_rag.py --use-gpu --output $OUT/4a_dense_generative.json \
  --retrieval dense --reader generative \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --andrew-id $ANDREW_ID --queries $Q

# 4b. Sparse-only
python run_rag.py --use-gpu --output $OUT/4b_sparse_generative.json \
  --retrieval sparse --reader generative \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --andrew-id $ANDREW_ID --queries $Q

# 4c. Hybrid (same as baseline, optional duplicate for table)
# python run_rag.py ... baseline already above

# ========== 5. Closed-book (no retrieval) ==========
python run_rag.py --use-gpu --output $OUT/5_closedbook_extractive.json \
  --reader extractive --closed-book \
  --andrew-id $ANDREW_ID --queries $Q

python run_rag.py --use-gpu --output $OUT/5_closedbook_generative.json \
  --reader generative --closed-book \
  --andrew-id $ANDREW_ID --queries $Q

# ========== 6. Sentence chunking (MiniLM + hybrid + generative) ==========
python run_rag.py --use-gpu --output $OUT/6_sentence_chunk_hybrid_generative.json \
  --retrieval hybrid --reader generative --chunker sentence \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --andrew-id $ANDREW_ID --queries $Q

# ========== 7. Paragraph chunking (MiniLM + hybrid + generative) ==========
python run_rag.py --use-gpu --output $OUT/7_paragraph_chunk_hybrid_generative.json \
  --retrieval hybrid --reader generative --chunker paragraph \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --andrew-id $ANDREW_ID --queries $Q

echo "Done. Outputs in $OUT/"
