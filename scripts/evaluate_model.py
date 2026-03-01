"""
MathStack - Universal evaluation script.
Loads GGUF via llama-cpp-python, runs 200 test problems, extracts first number,
scores exact match per type and overall. Saves results JSON.
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def load_config():
    import yaml
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_first_number(text: str):
    """Extract first number from model response (integer or decimal)."""
    if not text or not text.strip():
        return None
    # Match integers or decimals
    m = re.search(r"-?\d+\.?\d*", text.strip())
    return m.group(0) if m else None


def main():
    parser = argparse.ArgumentParser(description="Evaluate GGUF model on test set")
    parser.add_argument("--gguf", required=True, help="Path to .gguf model")
    parser.add_argument("--output", required=True, help="Path to save results JSON")
    parser.add_argument("--stage", type=str, default="eval", help="Label for this run")
    parser.add_argument("--use-rag", action="store_true", help="Use RAG-augmented prompts (stage 6)")
    args = parser.parse_args()

    config = load_config()
    test_path = PROJECT_ROOT / config["test_200"].replace("./", "")
    if not test_path.exists():
        print(f"Test file not found: {test_path}")
        sys.exit(1)

    # Load test problems
    problems = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    print(f"Loaded {len(problems)} test problems")

    # Optional RAG context builder
    if args.use_rag:
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            from rag.query_rag import build_augmented_prompt
            get_prompt = lambda p: build_augmented_prompt(p["instruction"], config)
        except Exception as e:
            print(f"RAG not available: {e}. Proceeding without RAG.")
            get_prompt = None
    else:
        get_prompt = None

    # Load LLM (llama-cpp-python with GPU layers)
    try:
        from llama_cpp import Llama
    except ImportError:
        print("llama-cpp-python not installed. pip install llama-cpp-python")
        sys.exit(1)

    print(f"Loading model: {args.gguf} (n_gpu_layers=99 for Vulkan)...")
    llm = Llama(model_path=args.gguf, n_ctx=512, n_gpu_layers=99, verbose=False)

    results = []
    prompt_template = "### Instruction: {instruction}\n\n### Response:"
    n_probs = len(problems)
    print(f"Evaluating 0/{n_probs}...", flush=True)

    for i, prob in enumerate(problems):
        instruction = prob["instruction"]
        correct_answer = prob["correct_answer"]
        ptype = prob["type"]

        if get_prompt:
            full_prompt = get_prompt(prob)
        else:
            full_prompt = prompt_template.format(instruction=instruction)

        out = llm(
            full_prompt,
            max_tokens=30,
            temperature=0.0,
            stop=["###", "\n\n"],
        )
        model_response = out["choices"][0]["text"].strip() if out.get("choices") else ""
        extracted = extract_first_number(model_response)
        # Compare as numbers if possible
        try:
            ext_num = int(float(extracted)) if extracted else None
        except (ValueError, TypeError):
            ext_num = None
        correct = ext_num == correct_answer if ext_num is not None else False

        results.append({
            "instruction": instruction,
            "correct_answer": correct_answer,
            "model_response": model_response,
            "extracted_number": extracted,
            "correct": correct,
            "type": ptype,
        })
        if (i + 1) % 20 == 0 or (i + 1) == n_probs:
            print(f"  Evaluated {i+1}/{n_probs}", flush=True)

    # Per-type accuracy
    type_correct = {}
    type_total = {}
    for r in results:
        t = r["type"]
        type_total[t] = type_total.get(t, 0) + 1
        if r["correct"]:
            type_correct[t] = type_correct.get(t, 0) + 1

    type_order = [
        "single_digit_addition", "single_digit_subtraction", "single_digit_multiplication",
        "double_digit_addition", "double_digit_subtraction", "double_digit_multiplication",
        "addition_with_carrying", "three_number_addition", "simple_division", "mixed_operations",
    ]
    print("")
    print("=" * 60)
    print(f"EVALUATION: {args.stage}")
    print("=" * 60)
    gaps = []
    for t in type_order:
        total = type_total.get(t, 0)
        correct = type_correct.get(t, 0)
        pct = (100.0 * correct / total) if total else 0
        short = t.replace("_", " ")[:28].ljust(28)
        print(f"  {short} {correct}/{total} = {pct:.0f}%")
        if total and pct < 50:
            gaps.append(t)
    overall_c = sum(r["correct"] for r in results)
    overall_t = len(results)
    overall_pct = 100.0 * overall_c / overall_t if overall_t else 0
    print("-" * 60)
    print(f"  OVERALL{' ':22} {overall_c}/{overall_t} = {overall_pct:.1f}%")
    print("=" * 60)
    if gaps:
        print("GAPS (below 50%): " + ", ".join(gaps))

    # Save
    out_data = {
        "stage": args.stage,
        "overall_accuracy": overall_pct,
        "overall_correct": overall_c,
        "overall_total": overall_t,
        "per_type": {t: {"correct": type_correct.get(t, 0), "total": type_total.get(t, 0)} for t in type_order},
        "gaps": gaps,
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
