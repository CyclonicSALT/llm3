#!/usr/bin/env bash
# MathStack - Full pipeline (bash, e.g. Kaggle / Linux).
# Saves progress: skips steps whose outputs already exist. Resume anytime.
set -e
cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "========================================"
echo "MATHSTACK - FULL PIPELINE"
echo "========================================"
echo "Project root: $PROJECT_ROOT"
echo ""

# Step 0: Generate datasets (skip if train_100 exists)
if [ ! -f "data/train_100.jsonl" ]; then
  echo "[Step 0] Generating arithmetic datasets..."
  python data/generate_arithmetic.py
  echo ""
else
  echo "[Step 0] data/train_100.jsonl exists. Skipping generate."
  echo ""
fi

# Step 1: Probe scores (needed for Stage 2). Train probe/cot if needed, then eval.
if [ ! -f "output/probe_scores.json" ]; then
  if [ -f "output/probe-q4.gguf" ]; then
    echo "[Step 1] Evaluating probe (GGUF)..."
    python scripts/evaluate_model.py --gguf output/probe-q4.gguf --output output/probe_scores.json --stage probe
  elif [ -d "models/probe" ] || [ -d "models/cot" ]; then
    echo "[Step 1] Evaluating probe (HuggingFace)..."
    MODEL="models/probe"
    [ -d "models/probe" ] || MODEL="models/cot"
    python scripts/evaluate_model_hf.py --model "$MODEL" --output output/probe_scores.json --stage probe
  else
    echo "[Step 1] Training probe (100 plain examples)..."
    python scripts/train_model.py --data data/train_100.jsonl --output models/probe --samples 100
    python scripts/evaluate_model_hf.py --model models/probe --output output/probe_scores.json --stage probe
  fi
  echo ""
else
  echo "[Step 1] output/probe_scores.json exists. Skipping."
  echo ""
fi

# Step 2: Standard baseline (optional, long). Skip by default; set RUN_BASELINE=1 to run.
if [ "${RUN_BASELINE:-0}" = "1" ] && [ ! -f "output/standard_baseline_scores.json" ]; then
  echo "[Step 2] Training standard baseline (1000 examples, ~4h)..."
  RESUME=""
  [ -d "models/standard_baseline/checkpoint-"* ] 2>/dev/null && RESUME="--resume"
  python scripts/train_model.py --data data/train_1000.jsonl --output models/standard_baseline --samples 1000 $RESUME
  python scripts/evaluate_model_hf.py --model models/standard_baseline --output output/standard_baseline_scores.json --stage standard_baseline
  echo ""
else
  echo "[Step 2] Skipping standard baseline (set RUN_BASELINE=1 to run)."
  echo ""
fi

# Step 3: Stage 1 - CoT
if [ ! -f "output/stage1_cot_scores.json" ]; then
  echo "[Step 3] Stage 1 - Chain of Thought..."
  python scripts/stage1_cot_format.py
  echo ""
else
  echo "[Step 3] output/stage1_cot_scores.json exists. Skipping Stage 1."
  echo ""
fi

# Step 4: Stage 2 - Probe guided
if [ ! -f "output/stage2_probe_guided_scores.json" ]; then
  echo "[Step 4] Stage 2 - Probe guided..."
  python scripts/stage2_probe_guided.py
  echo ""
else
  echo "[Step 4] output/stage2_probe_guided_scores.json exists. Skipping Stage 2."
  echo ""
fi

# Step 5: Stage 3 - MoE
if [ ! -f "output/stage3_moe_scores.json" ] && [ ! -d "models/moe/expert_addition" ]; then
  echo "[Step 5] Stage 3 - MoE..."
  python scripts/stage3_moe_train.py
  echo ""
else
  echo "[Step 5] MoE output exists. Skipping Stage 3."
  echo ""
fi

# Step 6: Stage 4 - Pruning
if [ ! -f "output/stage4_pruned_scores.json" ]; then
  echo "[Step 6] Stage 4 - Pruning..."
  python scripts/stage4_prune.py
  echo ""
else
  echo "[Step 6] Stage 4 output exists. Skipping."
  echo ""
fi

# Step 7: Stage 5 - QAT
if [ ! -f "output/stage5_qat_scores.json" ] && [ ! -d "models/final" ]; then
  echo "[Step 7] Stage 5 - QAT..."
  python scripts/stage5_qat_train.py
  echo ""
else
  echo "[Step 7] Final model exists. Skipping Stage 5."
  echo ""
fi

# Step 8: RAG index (optional)
if [ -f "rag/build_index.py" ] && [ ! -d "rag/chroma_db" ]; then
  echo "[Step 8] Building RAG index..."
  python rag/build_index.py 2>/dev/null || echo "RAG index skipped (optional)."
  echo ""
else
  echo "[Step 8] RAG index exists or script missing. Skipping."
  echo ""
fi

# Step 9: Stage 6 - RAG evaluation (optional)
if [ -f "output/mathstack-final-q4.gguf" ] || [ -d "models/final" ]; then
  echo "[Step 9] Stage 6 - RAG evaluation (if available)..."
  python scripts/stage6_rag_integrate.py 2>/dev/null || echo "Stage 6 skipped (GGUF/final model)."
  echo ""
else
  echo "[Step 9] Skipping Stage 6 (no final GGUF)."
  echo ""
fi

# Step 10: Report
echo "[Step 10] Compare stages..."
python scripts/compare_stages.py 2>/dev/null || true
echo ""
echo "Pipeline run complete. Check output/ for scores and reports."
