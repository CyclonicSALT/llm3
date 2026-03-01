# MathStack - Stacked Optimization Proof of Concept

## 1. The experiment

We test whether **6 stacked techniques on 100 training examples** can match a model trained on **1000 examples** using brute force. We use **arithmetic** because answers are objectively right or wrong, so we can measure exactly whether each technique helps.

## 2. The 6 techniques (simply explained)

| Technique | What it does |
|-----------|--------------|
| **Chain of Thought** | Show reasoning steps in every response; acts as working memory for small models. |
| **Probe Guided** | Use a small “probe” model’s failures to find which problem types need more data, then add targeted examples. |
| **MoE** | Four operation specialists (addition, subtraction, multiplication, division+mixed) instead of one generalist. |
| **Structured Pruning** | Remove non-math weights (e.g. 30% of FFN) so capacity focuses on arithmetic. |
| **QAT** | Quantization-aware training: adapt weights for 4-bit *before* quantizing to reduce accuracy loss. |
| **RAG** | Look up arithmetic rules from a vector DB instead of memorizing everything in weights. |

## 3. Why arithmetic?

- **Objective scoring** – no ambiguity; we compare the model’s numeric answer to the ground truth.
- **Fast to evaluate** – 200 test problems run in minutes.
- **Proof of concept** – if the stacking pipeline works here, the same ideas can be applied to the pentest domain (LLM4).

## 4. Prerequisites

- **LLM1 llama.cpp** must be built (convert + quantize tools used for GGUF export).
- **Python 3.10+** with a project-local venv.
- **16 GB RAM** (training is CPU-only; inference can use AMD GPU via Vulkan in llama.cpp).
- **Windows:** A C++ build toolchain (e.g. Visual Studio 2019 Build Tools with “Desktop development with C++”) is required to build `llama-cpp-python`. See **Troubleshooting** below if `pip install` fails with “No CMAKE_C_COMPILER could be found”.

## 5. How to run

```powershell
# One-time setup (venv, dependencies, directories)
.\setup.ps1

# Full pipeline (6–8 hours; baseline step ~4 hours)
.\run_pipeline.ps1

# Single stage (e.g. stage 3)
.\run_stage.ps1 -stage 3
```

Valid stages: `1`, `2`, `3`, `4`, `5`, `6`, `baseline`.

## 6. Reading the results

- **Key number:** stacked model accuracy (after all 6 stages, on 100 examples) vs standard baseline (1000 examples).
- **90%+ match** → concept proven; ready to apply to pentest.
- **75–90% match** → promising; refine weak stages.
- **Below 75%** → improve data or hyperparameters for failing types.

Reports:

- `output/stacking_report.json` – full comparison and verdict.
- `output/compression_framework.json` – gap analysis from the probe; reuse this logic for pentest.

## 7. Run on Kaggle (P100 GPU)

The project is set up to **use GPU only when running on Kaggle**. Locally it keeps using CPU.

**On Kaggle:**

1. **New Notebook** → Settings: **Internet** = On, **Accelerator** = **GPU (P100)**.
2. **Add your dataset** (right sidebar → “Add data” → pick your dataset). It will be at `/kaggle/input/<your-dataset-slug>/`.
3. **Clone the repo and wire in the data** (run in a notebook cell or in the notebook’s “Console” / terminal):
   ```bash
   git clone https://github.com/CyclonicSALT/llm3.git
   cd llm3
   # Replace YOUR_DATASET_SLUG with your dataset’s URL slug (e.g. my-llm3-data)
   cp -r /kaggle/input/YOUR_DATASET_SLUG/* data/  2>/dev/null || cp /kaggle/input/YOUR_DATASET_SLUG/*.jsonl data/
   pip install -r requirements.txt
   ```
4. **Run from project root** (so `device_utils` and `config.yaml` are found). Example – single training run:
   ```bash
   python scripts/train_model.py --data data/train_100.jsonl --output models/my_run
   ```
   On Kaggle (Linux) run the **Python scripts** directly; the full pipeline script is PowerShell for local use. Run one stage at a time, e.g. `python scripts/train_model.py --data data/train_100.jsonl --output models/cot`.
5. You should see e.g. `Using GPU: Tesla P100-PCIE-16GB [Kaggle]`.  
   **Note:** `config.yaml` uses Windows paths for llama.cpp (GGUF export); on Kaggle you can skip export or override those. Training and pruning run without them.

**Fall back to CPU on Kaggle:** if the GPU path causes issues, force CPU with:

```bash
export FORCE_CPU=1
python scripts/train_model.py ...
```

Same works in a notebook: `import os; os.environ["FORCE_CPU"] = "1"` before loading the model, then run as usual.

## 8. Troubleshooting (Windows)

### llama-cpp-python fails to build: “No CMAKE_C_COMPILER could be found”

CMake may pick a Visual Studio version that has no C++ compiler installed. Force a working generator before installing:

```powershell
$env:CMAKE_GENERATOR = "Visual Studio 16 2019"
$env:CMAKE_GENERATOR_PLATFORM = "x64"
.venv\Scripts\pip install llama-cpp-python
```

Then install the rest of the dependencies:

```powershell
.venv\Scripts\pip install -r requirements.txt
```

If you use a different Visual Studio version, set `CMAKE_GENERATOR` to its generator (e.g. `"Visual Studio 17 2022"` for VS 2022).

### “422 rules” instead of 500 for arithmetic_facts

The RAG facts generator caps at the number of facts it creates per category; 422 is fine and the pipeline works the same.

## 7. Next steps

- If the concept is proven: copy the **compression framework** and pipeline design from MathStack (Z:\Cursor\LLM3) to the **pentest domain** (Z:\Cursor\LLM4). The techniques that work on math will be applied there with pentest-specific data and evaluation.
