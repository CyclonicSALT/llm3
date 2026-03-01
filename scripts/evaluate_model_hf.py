"""
MathStack - Evaluate a HuggingFace model (or LoRA adapter) on the test set.
Writes the same JSON format as evaluate_model.py so Stage 2 can use it without GGUF.
Use this on Kaggle when llama.cpp export is not available.
"""

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config():
    import yaml
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_first_number(text: str):
    if not text or not text.strip():
        return None
    m = re.search(r"-?\d+\.?\d*", text.strip())
    return m.group(0) if m else None


def main():
    parser = argparse.ArgumentParser(description="Evaluate HF model on test set (same JSON as evaluate_model.py)")
    parser.add_argument("--model", required=True, help="Path to HuggingFace model or LoRA adapter (e.g. models/probe or models/cot)")
    parser.add_argument("--output", required=True, help="Path to save results JSON")
    parser.add_argument("--stage", type=str, default="eval", help="Label for this run")
    args = parser.parse_args()

    config = load_config()
    test_path = PROJECT_ROOT / config["test_200"].replace("./", "")
    if not test_path.exists():
        print(f"Test file not found: {test_path}")
        sys.exit(1)

    problems = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    print(f"Loaded {len(problems)} test problems")

    from device_utils import get_device_map, print_device_info
    print_device_info()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        model_path = PROJECT_ROOT / args.model
    if not model_path.exists():
        print(f"Model path not found: {args.model}")
        sys.exit(1)

    is_adapter = (model_path / "adapter_config.json").exists()
    if is_adapter:
        base_name = config["base_model"]
        cache = config.get("model_cache", "./models/base")
        print(f"Loading base {base_name} + adapter {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(base_name, cache_dir=cache, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_name, cache_dir=cache, trust_remote_code=True,
            torch_dtype="float32", device_map=get_device_map(), low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(model, str(model_path))
        model.eval()
    else:
        print(f"Loading model from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), trust_remote_code=True,
            torch_dtype="float32", device_map=get_device_map(), low_cpu_mem_usage=True,
        )
        model.eval()

    prompt_template = "### Instruction: {instruction}\n\n### Response:"
    type_order = [
        "single_digit_addition", "single_digit_subtraction", "single_digit_multiplication",
        "double_digit_addition", "double_digit_subtraction", "double_digit_multiplication",
        "addition_with_carrying", "three_number_addition", "simple_division", "mixed_operations",
    ]
    results = []
    n_probs = len(problems)
    print(f"Evaluating 0/{n_probs}...", flush=True)

    for i, prob in enumerate(problems):
        instruction = prob["instruction"]
        correct_answer = prob["correct_answer"]
        ptype = prob["type"]
        full_prompt = prompt_template.format(instruction=instruction)
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        with __import__("torch").no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        model_response = reply.split("###")[0].split("\n\n")[0].strip()
        extracted = extract_first_number(model_response)
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

    type_correct = {}
    type_total = {}
    for r in results:
        t = r["type"]
        type_total[t] = type_total.get(t, 0) + 1
        if r["correct"]:
            type_correct[t] = type_correct.get(t, 0) + 1

    overall_c = sum(r["correct"] for r in results)
    overall_t = len(results)
    overall_pct = 100.0 * overall_c / overall_t if overall_t else 0
    gaps = [t for t in type_order if type_total.get(t, 0) and (100.0 * type_correct.get(t, 0) / type_total[t]) < 50]

    print("")
    print("=" * 60)
    print(f"EVALUATION: {args.stage}")
    print("=" * 60)
    for t in type_order:
        total = type_total.get(t, 0)
        correct = type_correct.get(t, 0)
        pct = (100.0 * correct / total) if total else 0
        short = t.replace("_", " ")[:28].ljust(28)
        print(f"  {short} {correct}/{total} = {pct:.0f}%")
    print("-" * 60)
    print(f"  OVERALL{' ':22} {overall_c}/{overall_t} = {overall_pct:.1f}%")
    print("=" * 60)

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
