"""
MathStack - Compare all stages and print stacking report.
Loads all score files, prints table, verdict, and saves stacking_report.json.
"""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def load_config():
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        import yaml
        return yaml.safe_load(f)


def load_scores(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    config = load_config()
    output_dir = PROJECT_ROOT / config["output_dir"].replace("./", "")

    # Map stage -> (label, score file key, examples_used)
    stages = [
        ("Probe (plain 100)", "probe_scores.json", 100),
        ("+ Chain of Thought", "stage1_cot_scores.json", 100),
        ("+ Probe Guided", "stage2_probe_guided_scores.json", 100),
        ("+ MoE Experts", "stage3_moe_scores.json", 100),
        ("+ Pruning", "stage4_pruned_scores.json", 100),
        ("+ QAT", "stage5_qat_scores.json", 100),
        ("+ RAG", "stage6_rag_scores.json", 100),
    ]
    standard_baseline = ("Standard Baseline", "standard_baseline_scores.json", 1000)

    rows = []
    prev_acc = None
    for label, filename, examples in stages:
        path = output_dir / filename
        data = load_scores(path)
        acc = data.get("overall_accuracy", 0) if data else None
        vs_prev = "---"
        if acc is not None and prev_acc is not None:
            vs_prev = f"+{acc - prev_acc:.1f}%" if acc >= prev_acc else f"{acc - prev_acc:.1f}%"
        if acc is not None:
            prev_acc = acc
        acc_str = f"{acc:.1f}%" if acc is not None else "N/A"
        rows.append((label, examples, acc_str, vs_prev))

    std_path = output_dir / standard_baseline[1]
    std_data = load_scores(std_path)
    std_acc = std_data.get("overall_accuracy", 0) if std_data else None
    std_str = f"{std_acc:.1f}%" if std_acc is not None else "N/A"

    # Stacked final = last stage with data (RAG)
    stacked_acc = None
    for label, _, acc_str, _ in reversed(rows):
        if acc_str != "N/A":
            try:
                stacked_acc = float(acc_str.replace("%", ""))
                break
            except ValueError:
                pass

    print("")
    print("=" * 60)
    print("MATHSTACK - STACKING REPORT")
    print("=" * 60)
    print("")
    print("The key experiment:")
    if std_acc is not None:
        print(f"  Standard 0.5B (1000 examples): {std_acc:.1f}%   <- brute force baseline")
    if stacked_acc is not None:
        print(f"  Our stacked 0.5B (100 examples): {stacked_acc:.1f}%   <- did we match it with less?")
    print("")
    print("Stage by stage gains:")
    print("-" * 60)
    print(f"{'Stage':<22} | {'Examples':<10} | {'Test Accuracy':<12} | vs Previous")
    print("-" * 60)
    for label, ex, acc_str, vs_prev in rows:
        print(f"{label:<22} | {ex:<10} | {acc_str:<12} | {vs_prev}")
    print("-" * 60)
    print(f"{'Standard Baseline':<22} | {1000:<10} | {std_str:<12} | (10x more data)")
    print("=" * 60)
    print("")

    # Verdict
    if std_acc is not None and stacked_acc is not None:
        ratio = stacked_acc / std_acc if std_acc > 0 else 0
        if ratio >= 0.9:
            print("VERDICT: CONCEPT PROVEN")
            print("6 stacked techniques on 100 examples matches")
            print("brute force training on 1000 examples")
            print("10x more data efficient")
            print("Ready to apply to pentest domain")
        elif ratio >= 0.75:
            print("VERDICT: CONCEPT PARTIALLY PROVEN")
            print("Close but needs refinement")
            best = max(rows, key=lambda r: (float(r[2].replace("%", "")) if r[2] != "N/A" else -1))
            print(f"Best performing stage: {best[0]}")
        else:
            print("VERDICT: CONCEPT NEEDS WORK")
            weak = [r for r in rows if r[2] != "N/A" and float(r[2].replace("%", "")) < 50]
            if weak:
                print(f"Weakest stages: {[r[0] for r in weak]}")
            print("Recommendation: increase data for failing types or tune hyperparameters")
    else:
        print("VERDICT: Run full pipeline to get baseline and stacked scores.")

    # Most/least impactful
    gains = []
    prev = None
    for label, _, acc_str, _ in rows:
        if acc_str == "N/A":
            continue
        acc = float(acc_str.replace("%", ""))
        if prev is not None:
            gains.append((label.strip(), acc - prev))
        prev = acc
    if gains:
        best_stage = max(gains, key=lambda x: x[1])
        worst_stage = min(gains, key=lambda x: x[1])
        print("")
        print(f"Most impactful technique: {best_stage[0]} (+{best_stage[1]:.1f}%)")
        print(f"Least impactful technique: {worst_stage[0]} ({worst_stage[1]:+.1f}%)")

    print("")
    framework_path = output_dir / "compression_framework.json"
    print(f"Compression framework saved to: {framework_path}")
    print("Apply this framework to pentest domain: Z:\\Cursor\\LLM3 -> Z:\\Cursor\\LLM4")

    # Save report
    report = {
        "stages": [{"label": r[0], "examples": r[1], "accuracy": r[2], "vs_previous": r[3]} for r in rows],
        "standard_baseline": {"examples": 1000, "accuracy": std_str},
        "stacked_final_accuracy": stacked_acc,
        "standard_baseline_accuracy": std_acc,
    }
    report_path = output_dir / "stacking_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
