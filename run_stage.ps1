# MathStack - Run a single stage
# Usage: .\run_stage.ps1 -stage <1|2|3|4|5|6|baseline>

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("1","2","3","4","5","6","baseline")]
    [string]$stage
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot
.\.venv\Scripts\Activate.ps1

$startTime = Get-Date

switch ($stage) {
    "1" {
        Write-Host "Stage 1: Chain of Thought - format training data with reasoning steps, train, export, evaluate." -ForegroundColor Cyan
        python scripts/stage1_cot_format.py
    }
    "2" {
        Write-Host "Stage 2: Probe Guided - find failure types from probe, generate targeted examples, train on enhanced data." -ForegroundColor Cyan
        python scripts/stage2_probe_guided.py
    }
    "3" {
        Write-Host "Stage 3: MoE - train 4 expert LoRAs (addition, subtraction, multiplication, division+mixed), router, evaluate." -ForegroundColor Cyan
        python scripts/stage3_moe_train.py
    }
    "4" {
        Write-Host "Stage 4: Structured Pruning - prune 30% FFN weights, recovery fine-tune, export, evaluate." -ForegroundColor Cyan
        python scripts/stage4_prune.py
    }
    "5" {
        Write-Host "Stage 5: QAT - quantization aware training, then export Q4 GGUF, evaluate." -ForegroundColor Cyan
        python scripts/stage5_qat_train.py
    }
    "6" {
        Write-Host "Stage 6: RAG - build index from arithmetic facts, then RAG-augmented evaluation." -ForegroundColor Cyan
        python rag/build_index.py
        python scripts/stage6_rag_integrate.py
    }
    "baseline" {
        Write-Host "Baseline: Train on 1000 examples and evaluate (control group). Expected ~4 hours." -ForegroundColor Cyan
        python scripts/train_model.py --data data/train_1000.jsonl --output models/standard_baseline --samples 1000
        if ($LASTEXITCODE -eq 0) {
            python scripts/export_gguf.py --model models/standard_baseline --name standard_baseline
            python scripts/evaluate_model.py --gguf output/standard_baseline-q4.gguf --output output/standard_baseline_scores.json --stage standard_baseline
        }
    }
}

$elapsed = (Get-Date) - $startTime
Write-Host ""
Write-Host "Stage $stage completed in $($elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
