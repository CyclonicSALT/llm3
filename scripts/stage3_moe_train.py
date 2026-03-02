"""
MathStack - Stage 3: Mixture of Experts (MoE).
Train 4 specialist LoRA adapters (addition, subtraction, multiplication, division+mixed).
A router selects the right expert from question symbols. Same 0.5B base, 4x specialized knowledge.
"""

import json
import pickle
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Map problem type -> expert name
TYPE_TO_EXPERT = {
    "single_digit_addition": "addition",
    "double_digit_addition": "addition",
    "addition_with_carrying": "addition",
    "three_number_addition": "addition",
    "single_digit_subtraction": "subtraction",
    "double_digit_subtraction": "subtraction",
    "single_digit_multiplication": "multiplication",
    "double_digit_multiplication": "multiplication",
    "simple_division": "division_and_mixed",
    "mixed_operations": "division_and_mixed",
}

EXPERT_TO_TYPES = {
    "addition": ["single_digit_addition", "double_digit_addition", "addition_with_carrying", "three_number_addition"],
    "subtraction": ["single_digit_subtraction", "double_digit_subtraction"],
    "multiplication": ["single_digit_multiplication", "double_digit_multiplication"],
    "division_and_mixed": ["simple_division", "mixed_operations"],
}


def load_config():
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_features(instruction: str):
    """Simple symbol-based features for router."""
    return [
        1 if "+" in instruction else 0,
        1 if "-" in instruction else 0,
        1 if "x" in instruction or "*" in instruction else 0,
        1 if "/" in instruction else 0,
        1 if "(" in instruction else 0,
    ]


def main():
    config = load_config()
    train_1000_path = PROJECT_ROOT / config["train_1000"].replace("./", "")
    enhanced_train_path = PROJECT_ROOT / config["enhanced_train"].replace("./", "")
    moe_output = PROJECT_ROOT / config["moe_output"].replace("./", "")
    output_dir = PROJECT_ROOT / config["output_dir"].replace("./", "")
    samples_per_expert = config.get("stage3_moe_samples", 100)

    expert_names = ["addition", "subtraction", "multiplication", "division_and_mixed"]
    expert_label = {"addition": 0, "subtraction": 1, "multiplication": 2, "division_and_mixed": 3}

    print("Stage 3: MoE expert training")
    print("Training router on train_1000 (symbol-based)...")

    # Load train_1000 for router
    X_router, y_router = [], []
    with open(train_1000_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            X_router.append(extract_features(ex["instruction"]))
            y_router.append(expert_label[TYPE_TO_EXPERT[ex["type"]]])

    from sklearn.linear_model import LogisticRegression
    router = LogisticRegression(max_iter=500, random_state=42)
    router.fit(X_router, y_router)
    router_path = moe_output / "router.pkl"
    router_path.parent.mkdir(parents=True, exist_ok=True)
    with open(router_path, "wb") as f:
        pickle.dump(router, f)
    print(f"Router saved to {router_path}")

    # Load enhanced data and filter by expert
    enhanced = []
    if enhanced_train_path.exists():
        with open(enhanced_train_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    enhanced.append(json.loads(line))

    for expert in expert_names:
        types = EXPERT_TO_TYPES[expert]
        subset = [ex for ex in enhanced if ex.get("type") in types][:samples_per_expert]
        if not subset:
            print(f"No data for expert {expert}, skipping.")
            continue
        expert_dir = moe_output / f"expert_{expert}"
        expert_dir.mkdir(parents=True, exist_ok=True)
        data_path = expert_dir / "train_subset.jsonl"
        with open(data_path, "w", encoding="utf-8") as f:
            for ex in subset:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Training expert_{expert} on {len(subset)} examples...")
        subprocess.run([
            sys.executable,
            SCRIPT_DIR / "train_model.py",
            "--data", str(data_path),
            "--output", str(expert_dir),
            "--samples", str(len(subset)),
        ], cwd=PROJECT_ROOT, check=True)

    export_model = moe_output / "expert_addition"
    gguf_path = output_dir / "stage3-moe-q4.gguf"
    scores_path = output_dir / "stage3_moe_scores.json"
    sys.path.insert(0, str(PROJECT_ROOT))
    from device_utils import is_kaggle
    if not is_kaggle():
        print("Exporting MoE representative (expert_addition) to GGUF...")
        ret = subprocess.run([
            sys.executable, SCRIPT_DIR / "export_gguf.py",
            "--model", str(export_model), "--name", "stage3-moe",
        ], cwd=PROJECT_ROOT)
        if ret.returncode == 0 and gguf_path.exists():
            subprocess.run([
                sys.executable, SCRIPT_DIR / "evaluate_model.py",
                "--gguf", str(gguf_path), "--output", str(scores_path), "--stage", "stage3_moe",
            ], cwd=PROJECT_ROOT, check=True)
    if not scores_path.exists():
        print("Evaluating MoE (HuggingFace, no GGUF)...")
        subprocess.run([
            sys.executable, SCRIPT_DIR / "evaluate_model_hf.py",
            "--model", str(export_model), "--output", str(scores_path), "--stage", "stage3_moe",
        ], cwd=PROJECT_ROOT, check=True)
    if scores_path.exists():
        with open(scores_path, "r", encoding="utf-8") as f:
            acc = json.load(f).get("overall_accuracy", 0)
        print(f"Stage 3 MoE (representative expert) accuracy: {acc:.1f}%")
    print("Stage 3 complete. For full MoE routing use models/moe/moe_inference.py with router.pkl.")


if __name__ == "__main__":
    main()
