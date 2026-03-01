"""
MathStack - Stage 6: RAG integration.
Evaluate final GGUF with RAG-augmented prompts (relevant arithmetic rules as context).
"""

import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def load_config():
    import yaml
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_first_number(text):
    if not text or not text.strip():
        return None
    m = re.search(r"-?\d+\.?\d*", text.strip())
    return m.group(0) if m else None


def main():
    config = load_config()
    output_dir = PROJECT_ROOT / config["output_dir"].replace("./", "")
    final_gguf = output_dir / "mathstack-final-q4.gguf"
    test_path = PROJECT_ROOT / config["test_200"].replace("./", "")
    scores_path = output_dir / "stage6_rag_scores.json"

    if not final_gguf.exists():
        print("Final GGUF not found. Run stage 5 first.")
        sys.exit(1)

    sys.path.insert(0, str(PROJECT_ROOT))
    from rag.query_rag import build_augmented_prompt, query_rules

    with open(test_path, "r", encoding="utf-8") as f:
        problems = [json.loads(line) for line in f if line.strip()]
    n_probs = len(problems)
    print(f"Evaluating 0/{n_probs}...", flush=True)

    from llama_cpp import Llama
    llm = Llama(model_path=str(final_gguf), n_ctx=512, n_gpu_layers=99, verbose=False)

    results = []
    rule_usage = {}
    for i, prob in enumerate(problems):
        instruction = prob["instruction"]
        correct_answer = prob["correct_answer"]
        ptype = prob["type"]
        full_prompt = build_augmented_prompt(instruction, config)
        rules = query_rules(instruction, config.get("rag_top_k", 3))
        for r in rules:
            rule_usage[r[:80]] = rule_usage.get(r[:80], 0) + 1

        out = llm(full_prompt, max_tokens=30, temperature=0.0, stop=["###", "\n\n"])
        model_response = out["choices"][0]["text"].strip() if out.get("choices") else ""
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

    # Per-type and overall
    type_total = {}
    type_correct = {}
    for r in results:
        t = r["type"]
        type_total[t] = type_total.get(t, 0) + 1
        if r["correct"]:
            type_correct[t] = type_correct.get(t, 0) + 1
    overall_c = sum(r["correct"] for r in results)
    overall_t = len(results)
    overall_pct = 100.0 * overall_c / overall_t

    print("")
    print("Stage 6: RAG-augmented evaluation")
    print(f"OVERALL: {overall_c}/{overall_t} = {overall_pct:.1f}%")
    # Top rules that were retrieved most
    sorted_rules = sorted(rule_usage.items(), key=lambda x: -x[1])[:5]
    print("Most frequently retrieved rules (first 80 chars):")
    for r, c in sorted_rules:
        print(f"  {c}x: {r}...")

    out_data = {
        "stage": "stage6_rag",
        "overall_accuracy": overall_pct,
        "overall_correct": overall_c,
        "overall_total": overall_t,
        "per_type": {t: {"correct": type_correct.get(t, 0), "total": type_total.get(t, 0)} for t in type_total},
        "results": results,
    }
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {scores_path}")
    print("Stage 6 complete.")


if __name__ == "__main__":
    main()
