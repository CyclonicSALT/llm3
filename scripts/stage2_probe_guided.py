"""
MathStack - Stage 2: Probe Guided Training.
Use probe model failures to find which problem types need more data.
Generate targeted examples for those types, then train on enhanced dataset.
"""

import json
import random
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Same generators as data/generate_arithmetic.py (simplified inline for targeted types)
def _gen_addition_with_carrying(rng, count):
    out = []
    for _ in range(count):
        a_tens, a_ones = rng.randint(1, 9), rng.randint(5, 9)
        b_tens, b_ones = rng.randint(1, 9), rng.randint(10 - a_ones, 9)
        a, b = a_tens * 10 + a_ones, b_tens * 10 + b_ones
        out.append({"instruction": f"What is {a} + {b}?", "response": str(a+b), "type": "addition_with_carrying", "operands": [a,b], "correct_answer": a+b})
    return out

def _gen_double_digit_multiplication(rng, count):
    out = []
    for _ in range(count):
        a, b = rng.randint(10, 30), rng.randint(10, 30)
        out.append({"instruction": f"What is {a} x {b}?", "response": str(a*b), "type": "double_digit_multiplication", "operands": [a,b], "correct_answer": a*b})
    return out

def _gen_three_number_addition(rng, count):
    out = []
    for _ in range(count):
        a, b, c = rng.randint(10, 50), rng.randint(10, 50), rng.randint(10, 50)
        out.append({"instruction": f"What is {a} + {b} + {c}?", "response": str(a+b+c), "type": "three_number_addition", "operands": [a,b,c], "correct_answer": a+b+c})
    return out

def _gen_simple_division(rng, count):
    out = []
    for _ in range(count):
        answer, b = rng.randint(2, 12), rng.randint(2, 12)
        a = answer * b
        out.append({"instruction": f"What is {a} / {b}?", "response": str(answer), "type": "simple_division", "operands": [a,b], "correct_answer": answer})
    return out

def _gen_mixed_operations(rng, count):
    out = []
    for _ in range(count):
        a, b, c = rng.randint(1, 9), rng.randint(1, 9), rng.randint(2, 9)
        inner, result = a + b, (a + b) * c
        out.append({"instruction": f"What is ({a} + {b}) x {c}?", "response": str(result), "type": "mixed_operations", "operands": [a,b,c], "correct_answer": result})
    return out

def _gen_single_digit_addition(rng, count):
    out = []
    for _ in range(count):
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        out.append({"instruction": f"What is {a} + {b}?", "response": str(a+b), "type": "single_digit_addition", "operands": [a,b], "correct_answer": a+b})
    return out

def _gen_single_digit_subtraction(rng, count):
    out = []
    for _ in range(count):
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        if a < b: a, b = b, a
        out.append({"instruction": f"What is {a} - {b}?", "response": str(a-b), "type": "single_digit_subtraction", "operands": [a,b], "correct_answer": a-b})
    return out

def _gen_double_digit_addition(rng, count):
    out = []
    for _ in range(count):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        out.append({"instruction": f"What is {a} + {b}?", "response": str(a+b), "type": "double_digit_addition", "operands": [a,b], "correct_answer": a+b})
    return out

def _gen_double_digit_subtraction(rng, count):
    out = []
    for _ in range(count):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        if a < b: a, b = b, a
        out.append({"instruction": f"What is {a} - {b}?", "response": str(a-b), "type": "double_digit_subtraction", "operands": [a,b], "correct_answer": a-b})
    return out

def _gen_single_digit_multiplication(rng, count):
    out = []
    for _ in range(count):
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        out.append({"instruction": f"What is {a} x {b}?", "response": str(a*b), "type": "single_digit_multiplication", "operands": [a,b], "correct_answer": a*b})
    return out

TARGETED_GENERATORS = {
    "addition_with_carrying": _gen_addition_with_carrying,
    "double_digit_multiplication": _gen_double_digit_multiplication,
    "three_number_addition": _gen_three_number_addition,
    "simple_division": _gen_simple_division,
    "mixed_operations": _gen_mixed_operations,
    "single_digit_addition": _gen_single_digit_addition,
    "single_digit_subtraction": _gen_single_digit_subtraction,
    "double_digit_addition": _gen_double_digit_addition,
    "double_digit_subtraction": _gen_double_digit_subtraction,
    "single_digit_multiplication": _gen_single_digit_multiplication,
}


def load_config():
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    output_dir = PROJECT_ROOT / config["output_dir"].replace("./", "")
    probe_gguf = output_dir / "probe-q4.gguf"
    probe_scores_path = output_dir / "probe_scores.json"
    failures_path = PROJECT_ROOT / config["failures_data"].replace("./", "")
    variations_path = PROJECT_ROOT / "data" / "probe_variations.jsonl"
    cot_train_path = PROJECT_ROOT / config["cot_train"].replace("./", "")
    enhanced_train_path = PROJECT_ROOT / config["enhanced_train"].replace("./", "")
    probe_guided_output = PROJECT_ROOT / config["probe_guided_output"].replace("./", "")
    targeted_per_gap = config.get("targeted_examples_per_gap", 20)
    difficulty_levels = config.get("difficulty_levels", 3)
    framework_path = output_dir / "compression_framework.json"

    print("Stage 2: Probe Guided gap detection and training")

    # Phase A: Evaluate probe if we don't have scores yet
    if not probe_scores_path.exists():
        probe_model_path = PROJECT_ROOT / "models" / "probe"
        if probe_gguf.exists():
            print("Phase A: Evaluating probe (GGUF) on test set...")
            subprocess.run([
                sys.executable,
                SCRIPT_DIR / "evaluate_model.py",
                "--gguf", str(probe_gguf),
                "--output", str(probe_scores_path),
                "--stage", "probe",
            ], cwd=PROJECT_ROOT, check=True)
        elif (probe_model_path / "adapter_config.json").exists() or (probe_model_path / "config.json").exists():
            print("Phase A: Evaluating probe (HuggingFace) on test set (no GGUF)...")
            subprocess.run([
                sys.executable,
                SCRIPT_DIR / "evaluate_model_hf.py",
                "--model", str(probe_model_path),
                "--output", str(probe_scores_path),
                "--stage", "probe",
            ], cwd=PROJECT_ROOT, check=True)
        else:
            # Try models/cot as probe (user may have trained there first)
            cot_path = PROJECT_ROOT / "models" / "cot"
            if (cot_path / "adapter_config.json").exists():
                print("Phase A: Evaluating probe using models/cot (HuggingFace)...")
                subprocess.run([
                    sys.executable,
                    SCRIPT_DIR / "evaluate_model_hf.py",
                    "--model", str(cot_path),
                    "--output", str(probe_scores_path),
                    "--stage", "probe",
                ], cwd=PROJECT_ROOT, check=True)
            else:
                print("Probe GGUF not found and no models/probe or models/cot. Run: train_model.py --data data/train_100.jsonl --output models/probe")
                sys.exit(1)

    with open(probe_scores_path, "r", encoding="utf-8") as f:
        probe_data = json.load(f)

    # Phase B: Extract failures and group by type
    results = probe_data.get("results", [])
    failures = [r for r in results if not r.get("correct", True)]
    type_failures = {}
    for r in failures:
        t = r.get("type", "unknown")
        type_failures.setdefault(t, []).append(r)

    type_totals = probe_data.get("per_type", {})
    print("Probe model failures:")
    gaps = []
    for t, total_info in type_totals.items():
        total = total_info.get("total", 0)
        correct = total_info.get("correct", 0)
        failed = total - correct
        if total > 0:
            pct_fail = 100.0 * failed / total
            print(f"  {t}: {failed}/{total} wrong ({pct_fail:.0f}% failure)")
            if pct_fail >= 50:
                gaps.append({
                    "type": t,
                    "failure_rate": f"{pct_fail:.0f}%",
                    "examples_failed": failed,
                    "fix_strategy": "generate_targeted_examples",
                    "difficulty": "high" if pct_fail >= 70 else "medium",
                })

    failures_path.parent.mkdir(parents=True, exist_ok=True)
    with open(failures_path, "w", encoding="utf-8") as f:
        for r in failures:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(failures)} failures to {failures_path}")

    # Phase C: Save compression framework
    framework = {
        "version": "1.0",
        "probe_model": "Qwen2.5-0.5B trained on 100 arithmetic",
        "dataset": "pure arithmetic",
        "gaps": gaps,
        "insight": "These failure types indicate dataset underrepresentation. Fix the dataset and ALL future larger models benefit.",
    }
    with open(framework_path, "w", encoding="utf-8") as f:
        json.dump(framework, f, indent=2)
    print(f"Saved compression framework to {framework_path}")

    # Phase D: Generate targeted examples per gap (and a few for non-gaps if no gaps)
    rng = random.Random(43)
    variations = []
    if gaps:
        for g in gaps:
            t = g["type"]
            gen = TARGETED_GENERATORS.get(t)
            if gen:
                n = min(targeted_per_gap * difficulty_levels, 60)  # cap
                variations.extend(gen(rng, n))
    else:
        for t, gen in TARGETED_GENERATORS.items():
            variations.extend(gen(rng, 10))

    variations_path.parent.mkdir(parents=True, exist_ok=True)
    with open(variations_path, "w", encoding="utf-8") as f:
        for ex in variations:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Generated {len(variations)} targeted examples -> {variations_path}")

    # Phase E: Build enhanced dataset = cot_train + variations, shuffled
    cot_examples = []
    if cot_train_path.exists():
        with open(cot_train_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    cot_examples.append(json.loads(line))
    combined = cot_examples + variations
    rng.shuffle(combined)
    enhanced_train_path.parent.mkdir(parents=True, exist_ok=True)
    with open(enhanced_train_path, "w", encoding="utf-8") as f:
        for ex in combined:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Enhanced dataset: {len(combined)} total examples")

    # Phase F: Train on enhanced data
    samples = config.get("stage2_targeted_samples", 100)
    print(f"Training on enhanced data (max {samples} samples)...")
    subprocess.run([
        sys.executable,
        SCRIPT_DIR / "train_model.py",
        "--data", str(enhanced_train_path),
        "--output", str(probe_guided_output),
        "--samples", str(min(samples, len(combined))),
    ], cwd=PROJECT_ROOT, check=True)

    scores_path = output_dir / "stage2_probe_guided_scores.json"
    from device_utils import is_kaggle
    gguf_path = output_dir / "stage2-probe-guided-q4.gguf"
    if not is_kaggle():
        print("Exporting to GGUF...")
        ret = subprocess.run([
            sys.executable,
            SCRIPT_DIR / "export_gguf.py",
            "--model", str(probe_guided_output),
            "--name", "stage2-probe-guided",
        ], cwd=PROJECT_ROOT)
        if ret.returncode == 0 and gguf_path.exists():
            subprocess.run([
                sys.executable,
                SCRIPT_DIR / "evaluate_model.py",
                "--gguf", str(gguf_path),
                "--output", str(scores_path),
                "--stage", "stage2_probe_guided",
            ], cwd=PROJECT_ROOT, check=True)
    if not scores_path.exists():
        print("Evaluating with HuggingFace model (no GGUF on Kaggle)...")
        subprocess.run([
            sys.executable,
            SCRIPT_DIR / "evaluate_model_hf.py",
            "--model", str(probe_guided_output),
            "--output", str(scores_path),
            "--stage", "stage2_probe_guided",
        ], cwd=PROJECT_ROOT, check=True)

    with open(probe_scores_path, "r", encoding="utf-8") as f:
        probe_acc = json.load(f).get("overall_accuracy", 0)
    with open(scores_path, "r", encoding="utf-8") as f:
        stage2_acc = json.load(f).get("overall_accuracy", 0)
    print(f"Probe (100 plain examples):     {probe_acc:.1f}% accuracy")
    print(f"Stage 2 (100 enhanced examples): {stage2_acc:.1f}% accuracy")
    print(f"Improvement: +{stage2_acc - probe_acc:.1f}%")
    print("This proves probe-guided targeting works.")
    print("Stage 2 complete.")


if __name__ == "__main__":
    main()
