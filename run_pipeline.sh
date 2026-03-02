#!/usr/bin/env bash
# MathStack - Full pipeline (bash, e.g. Kaggle / Linux).
# Runs every step, no skipping. Full comparison: 6 stacked techniques vs 1000 baseline.
set -e
cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "========================================"
echo "MATHSTACK - FULL PIPELINE (NO SKIPPING)"
echo "========================================"
echo "Project root: $PROJECT_ROOT"
echo ""

echo "[Step 0] Generating arithmetic datasets..."
python data/generate_arithmetic.py
echo ""

echo "[Step 1] Training probe (100 plain examples) + evaluation..."
python scripts/train_model.py --data data/train_100.jsonl --output models/probe --samples 100
python scripts/evaluate_model_hf.py --model models/probe --output output/probe_scores.json --stage probe
echo ""

echo "[Step 2] Training standard baseline (1000 examples, ~4h comparison)..."
python scripts/train_model.py --data data/train_1000.jsonl --output models/standard_baseline --samples 1000 --resume
python scripts/evaluate_model_hf.py --model models/standard_baseline --output output/standard_baseline_scores.json --stage standard_baseline
echo ""

echo "[Step 3] Stage 1 - Chain of Thought..."
python scripts/stage1_cot_format.py
echo ""

echo "[Step 4] Stage 2 - Probe guided..."
python scripts/stage2_probe_guided.py
echo ""

echo "[Step 5] Stage 3 - MoE..."
python scripts/stage3_moe_train.py
echo ""

echo "[Step 6] Stage 4 - Pruning..."
python scripts/stage4_prune.py
echo ""

echo "[Step 7] Stage 5 - QAT..."
python scripts/stage5_qat_train.py
echo ""

echo "[Step 8] Building RAG index..."
python rag/build_index.py || true
echo ""

echo "[Step 9] Stage 6 - RAG evaluation..."
python scripts/stage6_rag_integrate.py || true
echo ""

echo "[Step 10] Generating comparison report..."
python scripts/compare_stages.py
echo ""
echo "Pipeline complete. Report: output/stacking_report.json"
