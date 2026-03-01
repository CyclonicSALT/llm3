"""
MathStack - Universal training script.
Used by all stages: loads base model, LoRA fine-tunes on JSONL, saves adapter.
CPU-only, fp32. Format: ### Instruction: ... ### Response: ...
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

# Add project root for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from device_utils import get_device_map, use_cpu, print_device_info


def load_config():
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_example(instruction: str, response: str) -> str:
    """Format one example for training (Alpaca-style)."""
    return f"### Instruction: {instruction}\n\n### Response: {response}"


def main():
    parser = argparse.ArgumentParser(description="Train model on arithmetic JSONL")
    parser.add_argument("--data", required=True, help="Path to training .jsonl")
    parser.add_argument("--output", required=True, help="Where to save model")
    parser.add_argument("--samples", type=int, default=None, help="Max samples (default: all)")
    parser.add_argument("--base", type=str, default=None, help="Base model path or HF name")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if exists")
    args = parser.parse_args()

    config = load_config()
    base_model = args.base or config["base_model"]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    examples = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    if args.samples is not None:
        examples = examples[: args.samples]
    print(f"Loaded {len(examples)} training examples")

    # Build dataset: instruction + response
    texts = []
    for ex in examples:
        instr = ex["instruction"]
        resp = ex["response"]
        texts.append(format_example(instr, resp))

    # Lazy imports for training stack
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    class ProgressCallback(TrainerCallback):
        """Print progress every time we log, so the terminal always shows movement."""
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_world_process_zero and logs and "loss" in logs:
                step, max_steps = state.global_step, state.max_steps
                loss = logs.get("loss", 0)
                print(f"  [Step {step}/{max_steps}] loss={loss:.4f}", flush=True)

    print_device_info()
    print(f"Loading base model: {base_model} (fp32)...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        cache_dir=config.get("model_cache", "./models/base"),
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        cache_dir=config.get("model_cache", "./models/base"),
        torch_dtype="float32",
        device_map=get_device_map(),
        low_cpu_mem_usage=True,
    )

    # LoRA: r=8, alpha=16, target all linear layers
    lora_config = LoraConfig(
        r=config.get("lora_r", 8),
        lora_alpha=config.get("lora_alpha", 16),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    dataset = Dataset.from_dict({"text": texts})

    # SFTConfig extends TrainingArguments and adds dataset_text_field, max_length (used by TRL 0.25+)
    max_seq = config.get("max_seq_length", 256)
    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.get("batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation", 16),
        num_train_epochs=config.get("epochs", 1),
        learning_rate=config.get("learning_rate", 2e-4),
        save_strategy=config.get("save_strategy", "steps"),
        save_steps=config.get("save_steps", 5),
        save_total_limit=config.get("save_total_limit", 2),
        logging_steps=1,
        use_cpu=use_cpu(),
        fp16=False,
        report_to=[],
        resume_from_checkpoint=bool(args.resume and any(output_dir.glob("checkpoint-*"))),
        dataset_text_field="text",
        max_length=max_seq,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[ProgressCallback()],
    )

    print("Starting training (progress below every step)...")
    start = time.perf_counter()
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    elapsed = time.perf_counter() - start
    print(f"Training finished in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save final adapter
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Log final loss if available
    if trainer.state.log_history:
        last_log = trainer.state.log_history[-1]
        if "loss" in last_log:
            print(f"Final loss: {last_log['loss']:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
