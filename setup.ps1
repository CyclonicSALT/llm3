# MathStack - One-time setup
# Creates venv, installs dependencies, verifies paths

Write-Host "================================" -ForegroundColor Cyan
Write-Host "MATHSTACK - SETUP" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

# 1. Create .venv
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor Green
}

# 2. Activate and upgrade pip
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 3. Install requirements
# On Windows, llama-cpp-python needs a valid CMake generator (e.g. VS 2019) or the build fails with "No CMAKE_C_COMPILER could be found"
if ($env:OS -eq "Windows_NT" -and -not $env:CMAKE_GENERATOR) {
    $env:CMAKE_GENERATOR = "Visual Studio 16 2019"
    $env:CMAKE_GENERATOR_PLATFORM = "x64"
    Write-Host "Set CMAKE_GENERATOR for llama-cpp-python build (Windows)." -ForegroundColor Gray
}
Write-Host "Installing dependencies (this may take several minutes)..." -ForegroundColor Yellow
pip install -r requirements.txt

# 4. Verify critical packages
Write-Host "Verifying packages..." -ForegroundColor Yellow
$packages = @("torch", "transformers", "peft", "trl", "chromadb", "sentence_transformers", "llama_cpp", "optimum", "sklearn")
foreach ($p in $packages) {
    $ok = python -c "import $($p.Replace('-','_')); print('ok')" 2>$null
    if ($ok -eq "ok") { Write-Host "  $p OK" -ForegroundColor Green } else { Write-Host "  $p MISSING" -ForegroundColor Red }
}

# 5. Check llama.cpp binaries
Write-Host "Checking llama.cpp tools..." -ForegroundColor Yellow
$llamaCli = "Z:\Cursor\LLM1\llama.cpp\build\bin\Release\llama-cli.exe"
$llamaQuant = "Z:\Cursor\LLM1\llama.cpp\build\bin\Release\llama-quantize.exe"
$llamaConvert = "Z:\Cursor\LLM1\llama.cpp\convert_hf_to_gguf.py"
if (Test-Path $llamaCli) { Write-Host "  llama-cli OK" -ForegroundColor Green } else { Write-Host "  llama-cli NOT FOUND" -ForegroundColor Red }
if (Test-Path $llamaQuant) { Write-Host "  llama-quantize OK" -ForegroundColor Green } else { Write-Host "  llama-quantize NOT FOUND" -ForegroundColor Red }
if (Test-Path $llamaConvert) { Write-Host "  convert_hf_to_gguf.py OK" -ForegroundColor Green } else { Write-Host "  convert_hf_to_gguf.py NOT FOUND" -ForegroundColor Red }

# 6. Create project directories
$dirs = @(
    "data", "data/rag_documents",
    "rag", "rag/chroma_db",
    "models", "models/probe", "models/standard_baseline", "models/cot",
    "models/probe_guided", "models/moe", "models/moe/expert_addition",
    "models/moe/expert_subtraction", "models/moe/expert_multiplication",
    "models/moe/expert_division", "models/pruned", "models/final",
    "models/base", "output"
)
foreach ($d in $dirs) {
    if (-not (Test-Path $d)) {
        New-Item -ItemType Directory -Path $d -Force | Out-Null
        Write-Host "  Created $d" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Run .\run_pipeline.ps1 to start the full pipeline." -ForegroundColor Cyan
Write-Host ""
Write-Host "WARNING: Standard baseline (step 2) takes ~4 hours. Consider running overnight or skipping for quick test." -ForegroundColor Yellow
