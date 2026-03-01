"""
MathStack - Stage 4: Structured Pruning.
Remove ~30% of FFN weights that contribute least (magnitude pruning).
Never prune attention, embeddings, or layer norms. Fine-tune briefly to recover.
"""

import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from device_utils import get_device_map, use_cpu, print_device_info


def load_config():
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_nonzero(model):
    return sum((p != 0).sum().item() for p in model.parameters())


def main():
    config = load_config()
    # Best model so far: stage 3 output (use probe_guided if MoE export is same as one expert)
    # We use probe_guided as the "best so far" for pruning input to keep pipeline simple
    probe_guided_path = PROJECT_ROOT / config["probe_guided_output"].replace("./", "")
    cot_path = PROJECT_ROOT / config["cot_output"].replace("./", "")
    # Prefer stage3 expert if available, else probe_guided
    moe_expert = PROJECT_ROOT / config["moe_output"].replace("./", "") / "expert_addition"
    if (moe_expert / "adapter_config.json").exists():
        load_path = moe_expert
    elif (probe_guided_path / "adapter_config.json").exists():
        load_path = probe_guided_path
    else:
        load_path = cot_path

    pruned_output = PROJECT_ROOT / config["pruned_output"].replace("./", "")
    enhanced_train_path = PROJECT_ROOT / config["enhanced_train"].replace("./", "")
    output_dir = PROJECT_ROOT / config["output_dir"].replace("./", "")
    sparsity = config.get("pruning_target_sparsity", 0.3)
    preserve_attention = config.get("preserve_attention", True)

    print("Stage 4: Structured pruning (FFN only, 30% sparsity)")
    print_device_info()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model_name = config["base_model"]
    cache = config.get("model_cache", "./models/base")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=cache, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        cache_dir=cache,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map=get_device_map(),
        low_cpu_mem_usage=True,
    )
    if (load_path / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(load_path))
        model = model.merge_and_unload()

    before_params = count_parameters(model)
    before_nonzero = count_nonzero(model)

    # Magnitude pruning: FFN layers only (gate_proj, up_proj, down_proj)
    prunable_names = ["gate_proj", "up_proj", "down_proj"] if preserve_attention else []
    n_pruned = 0
    for name, param in model.named_parameters():
        if not any(p in name for p in prunable_names):
            continue
        if param.dim() < 2:
            continue
        with torch.no_grad():
            flat = param.data.abs().flatten()
            k = max(1, int(flat.numel() * sparsity))
            thresh = torch.kthvalue(flat, k).values.item()
            mask = param.data.abs() >= thresh
            param.data.mul_(mask)
            n_pruned += (~mask).sum().item()

    after_nonzero = count_nonzero(model)
    print(f"Weights before pruning: {before_params / 1e6:.2f}M params")
    print(f"Weights zeroed: {n_pruned / 1e6:.2f}M ({100.0 * n_pruned / before_params:.1f}%)")
    print(f"Weights remaining active: {after_nonzero / 1e6:.2f}M")

    # Save pruned model then run brief fine-tune (100 steps)
    pruned_output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(pruned_output))
    tokenizer.save_pretrained(str(pruned_output))

    # Brief recovery fine-tune
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    class ProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_world_process_zero and logs and "loss" in logs:
                print(f"  [Step {state.global_step}/{state.max_steps}] loss={logs.get('loss', 0):.4f}", flush=True)

    examples = []
    if enhanced_train_path.exists():
        with open(enhanced_train_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ex = json.loads(line)
                    examples.append(f"### Instruction: {ex['instruction']}\n\n### Response: {ex['response']}")
    examples = examples[: config.get("stage4_prune_samples", 100)]
    dataset = Dataset.from_dict({"text": examples})

    training_args = SFTConfig(
        output_dir=str(pruned_output / "recovery"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_steps=100,
        learning_rate=1e-5,
        max_length=256,
        save_strategy="no",
        use_cpu=use_cpu(),
        report_to=[],
        dataset_text_field="text",
        packing=False,
        logging_steps=1,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[ProgressCallback()],
    )
    print("Recovery fine-tune (progress below)...")
    trainer.train()
    trainer.save_model(str(pruned_output))
    tokenizer.save_pretrained(str(pruned_output))

    print("Exporting pruned model to GGUF...")
    subprocess.run([
        sys.executable,
        SCRIPT_DIR / "export_gguf.py",
        "--model", str(pruned_output),
        "--name", "stage4-pruned",
    ], cwd=PROJECT_ROOT, check=True)

    scores_path = output_dir / "stage4_pruned_scores.json"
    gguf_path = output_dir / "stage4-pruned-q4.gguf"
    if gguf_path.exists():
        subprocess.run([
            sys.executable,
            SCRIPT_DIR / "evaluate_model.py",
            "--gguf", str(gguf_path),
            "--output", str(scores_path),
            "--stage", "stage4_pruned",
        ], cwd=PROJECT_ROOT, check=True)
        with open(scores_path, "r", encoding="utf-8") as f:
            acc = json.load(f).get("overall_accuracy", 0)
        print(f"Score after pruning + recovery: {acc:.1f}%")
    print("Stage 4 complete.")


if __name__ == "__main__":
    main()
