"""
MathStack - Export HuggingFace model to GGUF (fp16 then Q4_K_M).
If the model path is a LoRA adapter (no full config.json), we merge it into the base
model first, then convert. Uses llama.cpp from LLM1.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from device_utils import get_device_map, print_device_info


def load_config():
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Export HF model to GGUF Q4_K_M")
    parser.add_argument("--model", required=True, help="Path to HuggingFace model or LoRA adapter folder")
    parser.add_argument("--name", required=True, help="Output name (e.g. stage1-cot)")
    args = parser.parse_args()

    config = load_config()
    model_path = Path(args.model).resolve()
    if not model_path.exists():
        print(f"Model path not found: {model_path}")
        sys.exit(1)

    output_dir = PROJECT_ROOT / config.get("output_dir", "output").replace("./", "")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_model_name = config["base_model"]
    model_cache = config.get("model_cache", "./models/base")
    llama_convert = config["llama_convert"]
    llama_quantize = config["llama_quantize"]

    # Decide if we have a LoRA adapter (need to merge) or a full model (convert directly)
    is_lora = (model_path / "adapter_config.json").exists()
    has_full_config = (model_path / "config.json").exists()
    merged_dir = None

    if is_lora or not has_full_config:
        # LoRA adapter: merge into base, save to temp folder, then convert the merged folder
        merged_dir = output_dir / f"merged_{args.name}"
        merged_dir.mkdir(parents=True, exist_ok=True)
        try:
            print("Step 0: Merging LoRA adapter into base model...", flush=True)
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            print_device_info()
            print(f"  Loading base model: {base_model_name}", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                cache_dir=model_cache,
            )
            base = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                cache_dir=model_cache,
                torch_dtype="float32",
                device_map=get_device_map(),
                low_cpu_mem_usage=True,
            )
            print(f"  Loading adapter: {model_path}", flush=True)
            model = PeftModel.from_pretrained(base, str(model_path))
            print("  Merging adapter into base...", flush=True)
            merged = model.merge_and_unload()
            print(f"  Saving merged model to {merged_dir}...", flush=True)
            merged.save_pretrained(str(merged_dir))
            tokenizer.save_pretrained(str(merged_dir))
            convert_source = merged_dir
        except Exception as e:
            print(f"Merge failed: {e}")
            if merged_dir.exists():
                shutil.rmtree(merged_dir, ignore_errors=True)
            sys.exit(1)
    else:
        convert_source = model_path

    fp16_path = output_dir / f"{args.name}-fp16.gguf"
    q4_path = output_dir / f"{args.name}-q4.gguf"

    # 1. Convert to fp16 GGUF (from merged or full model folder)
    print(f"Step 1/3: Converting to fp16 GGUF... (this may take a few minutes)", flush=True)
    cmd_convert = [
        sys.executable,
        llama_convert,
        str(convert_source),
        "--outfile", str(fp16_path),
    ]
    ret = subprocess.run(cmd_convert, cwd=PROJECT_ROOT)
    if ret.returncode != 0:
        print("Conversion failed.")
        if merged_dir is not None and merged_dir.exists():
            shutil.rmtree(merged_dir, ignore_errors=True)
        sys.exit(1)

    # 2. Quantize to Q4_K_M
    print(f"Step 2/3: Quantizing to Q4_K_M...", flush=True)
    ret = subprocess.run([
        llama_quantize,
        str(fp16_path),
        str(q4_path),
        "Q4_K_M",
    ], cwd=PROJECT_ROOT)
    if ret.returncode != 0:
        print("Quantization failed.")
        if merged_dir is not None and merged_dir.exists():
            shutil.rmtree(merged_dir, ignore_errors=True)
        sys.exit(1)

    # 3. Clean up: delete fp16 GGUF and temp merged folder to save space
    print("Step 3/3: Cleaning up...", flush=True)
    try:
        fp16_path.unlink()
        print("  Removed fp16 GGUF.")
    except OSError:
        pass
    if merged_dir is not None and merged_dir.exists():
        try:
            shutil.rmtree(merged_dir)
            print(f"  Removed temp merged folder: {merged_dir.name}")
        except OSError as e:
            print(f"  Could not remove {merged_dir}: {e}")

    size_mb = q4_path.stat().st_size / (1024 * 1024)
    print(f"Final GGUF: {q4_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
