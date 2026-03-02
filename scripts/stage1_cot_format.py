"""
MathStack - Stage 1: Chain of Thought.
Reformat training data so every response shows explicit reasoning steps.
Small models score higher when they reason step-by-step (working memory).
"""

import json
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def load_config():
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cot_response(example: dict) -> str:
    """Build CoT response from problem type and operands."""
    ptype = example["type"]
    operands = example.get("operands", [])
    answer = example["correct_answer"]
    instr = example["instruction"]

    if ptype == "single_digit_addition":
        a, b = operands[0], operands[1]
        return f"Let me calculate. {a} + {b} = {answer}"

    if ptype == "single_digit_subtraction":
        a, b = operands[0], operands[1]
        return f"Let me calculate. {a} - {b} = {answer}"

    if ptype == "single_digit_multiplication":
        a, b = operands[0], operands[1]
        return f"Let me calculate. {a} x {b} = {answer}"

    if ptype == "double_digit_addition":
        a, b = operands[0], operands[1]
        a_ones, a_tens = a % 10, a // 10
        b_ones, b_tens = b % 10, b // 10
        ones_sum = a_ones + b_ones
        carry = 1 if ones_sum >= 10 else 0
        ones_digit = ones_sum % 10
        tens_sum = a_tens + b_tens + carry
        carry_text = ". Carry 1." if carry else "."
        plus_carry = " + 1 (carried)" if carry else ""
        return (
            f"Let me work column by column.\n"
            f"Ones: {a_ones} + {b_ones} = {ones_sum}. Write {ones_digit}{carry_text}\n"
            f"Tens: {a_tens} + {b_tens}{plus_carry} = {tens_sum}.\n"
            f"Result: {answer}"
        )

    if ptype == "double_digit_subtraction":
        a, b = operands[0], operands[1]
        return f"Let me work through this. {a} - {b} = {answer}"

    if ptype == "addition_with_carrying":
        a, b = operands[0], operands[1]
        a_ones, a_tens = a % 10, a // 10
        b_ones, b_tens = b % 10, b // 10
        ones_sum = a_ones + b_ones
        carry = 1
        ones_digit = ones_sum % 10
        tens_sum = a_tens + b_tens + carry
        return (
            f"Let me work through this step by step.\n"
            f"Ones column: {a_ones} + {b_ones} = {ones_sum}. Write {ones_digit}, carry 1.\n"
            f"Tens column: {a_tens} + {b_tens} + 1 (carried) = {tens_sum}.\n"
            f"Result: {answer}"
        )

    if ptype == "double_digit_multiplication":
        a, b = operands[0], operands[1]
        b_ones, b_tens = b % 10, b // 10
        partial1 = a * b_ones
        partial2 = a * b_tens * 10
        return (
            f"Let me break this down.\n"
            f"{a} x {b_ones} = {partial1}\n"
            f"{a} x {b_tens}0 = {partial2}\n"
            f"{partial1} + {partial2} = {answer}"
        )

    if ptype == "three_number_addition":
        a, b, c = operands[0], operands[1], operands[2]
        ab = a + b
        return (
            f"Adding left to right.\n"
            f"{a} + {b} = {ab}\n"
            f"{ab} + {c} = {answer}"
        )

    if ptype == "simple_division":
        a, b = operands[0], operands[1]
        return (
            f"Let me think. {a} / {b} = ?\n"
            f"{b} x {answer} = {a}.\n"
            f"So {a} / {b} = {answer}"
        )

    if ptype == "mixed_operations":
        a, b, c = operands[0], operands[1], operands[2]
        inner = a + b
        return (
            f"Brackets first.\n"
            f"({a} + {b}) = {inner}\n"
            f"{inner} x {c} = {answer}"
        )

    # Fallback
    return str(answer)


def main():
    config = load_config()
    train_100_path = PROJECT_ROOT / config["train_100"].replace("./", "")
    cot_train_path = PROJECT_ROOT / config["cot_train"].replace("./", "")
    cot_output = PROJECT_ROOT / config["cot_output"].replace("./", "")
    output_dir = PROJECT_ROOT / config["output_dir"].replace("./", "")

    print("Stage 1: Chain of Thought formatting and training")
    print("Generating CoT versions of train_100...")

    examples = []
    with open(train_100_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    cot_examples = []
    for ex in examples:
        cot_examples.append({
            "instruction": ex["instruction"],
            "response": cot_response(ex),
            "type": ex["type"],
            "operands": ex.get("operands", []),
            "correct_answer": ex["correct_answer"],
        })

    cot_train_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cot_train_path, "w", encoding="utf-8") as f:
        for ex in cot_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Saved {len(cot_examples)} examples to {cot_train_path}")

    # Train on CoT data
    samples = config.get("stage1_cot_samples", 100)
    print(f"Training model on CoT data ({samples} samples)...")
    ret = subprocess.run([
        sys.executable,
        SCRIPT_DIR / "train_model.py",
        "--data", str(cot_train_path),
        "--output", str(cot_output),
        "--samples", str(samples),
    ], cwd=PROJECT_ROOT)
    if ret.returncode != 0:
        sys.exit(1)

    # Export GGUF (skip on Kaggle; use HF eval instead)
    scores_path = output_dir / "stage1_cot_scores.json"
    sys.path.insert(0, str(PROJECT_ROOT))
    from device_utils import is_kaggle
    gguf_path = output_dir / "stage1-cot-q4.gguf"
    if not is_kaggle():
        print("Exporting to GGUF...")
        ret = subprocess.run([
            sys.executable,
            SCRIPT_DIR / "export_gguf.py",
            "--model", str(cot_output),
            "--name", "stage1-cot",
        ], cwd=PROJECT_ROOT)
        if ret.returncode == 0 and gguf_path.exists():
            print("Evaluating on test set (GGUF)...")
            ret = subprocess.run([
                sys.executable,
                SCRIPT_DIR / "evaluate_model.py",
                "--gguf", str(gguf_path),
                "--output", str(scores_path),
                "--stage", "stage1_cot",
            ], cwd=PROJECT_ROOT)
    if not scores_path.exists():
        print("Evaluating on test set (HuggingFace, no GGUF)...")
        ret = subprocess.run([
            sys.executable,
            SCRIPT_DIR / "evaluate_model_hf.py",
            "--model", str(cot_output),
            "--output", str(scores_path),
            "--stage", "stage1_cot",
        ], cwd=PROJECT_ROOT)
        if ret.returncode != 0:
            sys.exit(1)

    # Load probe scores for comparison (if available)
    probe_path = output_dir / "probe_scores.json"
    if probe_path.exists():
        with open(probe_path, "r", encoding="utf-8") as f:
            probe_data = json.load(f)
        with open(scores_path, "r", encoding="utf-8") as f:
            cot_data = json.load(f)
        probe_acc = probe_data.get("overall_accuracy", 0)
        cot_acc = cot_data.get("overall_accuracy", 0)
        diff = cot_acc - probe_acc
        print(f"Probe baseline: {probe_acc:.1f}%")
        print(f"Stage 1 (CoT):   {cot_acc:.1f}%")
        print(f"Improvement:    +{diff:.1f}%")
    else:
        print("Probe scores not found; run probe evaluation first for comparison.")

    print("Stage 1 complete. Scores saved to", scores_path)


if __name__ == "__main__":
    main()
