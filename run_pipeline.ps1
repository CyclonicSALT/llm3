# MathStack - Full pipeline (all stages)
# Estimated time: 6-8 hours total. Standard baseline alone ~4 hours.

Write-Host "================================" -ForegroundColor Cyan
Write-Host "MATHSTACK - FULL PIPELINE" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Testing 6 stacked techniques on pure arithmetic"
Write-Host "100 training examples vs 1000 brute force baseline"
Write-Host "Estimated time: 6-8 hours total"
Write-Host ""

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot
.\.venv\Scripts\Activate.ps1

$startTime = Get-Date

# Step 0: Generate datasets
Write-Host "Step 0: Generating arithmetic datasets..." -ForegroundColor Yellow
python data/generate_arithmetic.py
if ($LASTEXITCODE -ne 0) { exit 1 }
Write-Host ""

# Step 1: Probe model (skip if already exported)
if (Test-Path "output/probe-q4.gguf") {
    Write-Host "Step 1: Probe already trained and exported. Skipping training." -ForegroundColor Green
    if (-not (Test-Path "output/probe_scores.json")) {
        Write-Host "  Running probe evaluation..."
        python scripts/evaluate_model.py --gguf output/probe-q4.gguf --output output/probe_scores.json --stage probe
        if ($LASTEXITCODE -ne 0) { exit 1 }
    } else {
        Write-Host "  Probe scores already exist. Skipping." -ForegroundColor Gray
    }
} else {
    Write-Host "Step 1: Training probe model (100 plain examples)..." -ForegroundColor Yellow
    Write-Host "Expected: ~25 minutes"
    python scripts/train_model.py --data data/train_100.jsonl --output models/probe --samples 100
    if ($LASTEXITCODE -ne 0) { exit 1 }
    python scripts/export_gguf.py --model models/probe --name probe
    if ($LASTEXITCODE -ne 0) { exit 1 }
    python scripts/evaluate_model.py --gguf output/probe-q4.gguf --output output/probe_scores.json --stage probe
    if ($LASTEXITCODE -ne 0) { exit 1 }
}
Write-Host ""

# Step 2: Standard baseline (optional long run; resumes from checkpoint if exists)
if (Test-Path "output/standard_baseline-q4.gguf") {
    Write-Host "Step 2: Standard baseline already trained and exported. Skipping." -ForegroundColor Green
    if (-not (Test-Path "output/standard_baseline_scores.json")) {
        python scripts/evaluate_model.py --gguf output/standard_baseline-q4.gguf --output output/standard_baseline_scores.json --stage standard_baseline
        if ($LASTEXITCODE -ne 0) { exit 1 }
    }
} else {
    Write-Host "Step 2: Training standard baseline (1000 examples)..." -ForegroundColor Yellow
    Write-Host "Expected: ~4 hours (control group). Checkpoints every 5 steps - safe to close and resume with --resume later."
    $resumeArg = ""
    if (Get-ChildItem -Path "models/standard_baseline" -Filter "checkpoint-*" -Directory -ErrorAction SilentlyContinue) {
        $resumeArg = "--resume"
        Write-Host "  Resuming from existing checkpoint." -ForegroundColor Gray
    }
    python scripts/train_model.py --data data/train_1000.jsonl --output models/standard_baseline --samples 1000 $resumeArg
    if ($LASTEXITCODE -ne 0) { exit 1 }
    python scripts/export_gguf.py --model models/standard_baseline --name standard_baseline
    if ($LASTEXITCODE -ne 0) { exit 1 }
    python scripts/evaluate_model.py --gguf output/standard_baseline-q4.gguf --output output/standard_baseline_scores.json --stage standard_baseline
    if ($LASTEXITCODE -ne 0) { exit 1 }
}
Write-Host ""

# Step 3: Stage 1 - CoT
Write-Host "Step 3: Stage 1 - Chain of Thought formatting and training..." -ForegroundColor Yellow
Write-Host "Expected: ~30 minutes"
python scripts/stage1_cot_format.py
if ($LASTEXITCODE -ne 0) { exit 1 }
Write-Host ""

# Step 4: Stage 2 - Probe guided
Write-Host "Step 4: Stage 2 - Probe guided gap detection and training..." -ForegroundColor Yellow
Write-Host "Expected: ~30 minutes"
python scripts/stage2_probe_guided.py
if ($LASTEXITCODE -ne 0) { exit 1 }
Write-Host ""

# Step 5: Stage 3 - MoE
Write-Host "Step 5: Stage 3 - MoE expert training..." -ForegroundColor Yellow
Write-Host "Expected: ~1 hour (4 experts)"
python scripts/stage3_moe_train.py
if ($LASTEXITCODE -ne 0) { exit 1 }
Write-Host ""

# Step 6: Stage 4 - Pruning
Write-Host "Step 6: Stage 4 - Structured pruning..." -ForegroundColor Yellow
Write-Host "Expected: ~30 minutes"
python scripts/stage4_prune.py
if ($LASTEXITCODE -ne 0) { exit 1 }
Write-Host ""

# Step 7: Stage 5 - QAT
Write-Host "Step 7: Stage 5 - Quantization aware training..." -ForegroundColor Yellow
Write-Host "Expected: ~30 minutes"
python scripts/stage5_qat_train.py
if ($LASTEXITCODE -ne 0) { exit 1 }
Write-Host ""

# Step 8: RAG index
Write-Host "Step 8: Stage 6 - Building RAG index..." -ForegroundColor Yellow
Write-Host "Expected: ~5 minutes"
python rag/build_index.py
if ($LASTEXITCODE -ne 0) { exit 1 }
Write-Host ""

# Step 9: RAG evaluation
Write-Host "Step 9: Stage 6 - RAG augmented evaluation..." -ForegroundColor Yellow
Write-Host "Expected: ~10 minutes"
python scripts/stage6_rag_integrate.py
if ($LASTEXITCODE -ne 0) { exit 1 }
Write-Host ""

# Step 10: Report
Write-Host "Step 10: Generating stacking report..." -ForegroundColor Yellow
python scripts/compare_stages.py
if ($LASTEXITCODE -ne 0) { exit 1 }

$elapsed = (Get-Date) - $startTime
Write-Host ""
Write-Host "Total elapsed time: $($elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Cyan
Write-Host "Final model: output/mathstack-final-q4.gguf" -ForegroundColor Cyan
Write-Host "Report: output/stacking_report.json" -ForegroundColor Cyan
