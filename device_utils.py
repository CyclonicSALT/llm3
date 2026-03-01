"""
Device selection: use GPU only when running on Kaggle (e.g. P100).
Everywhere else, or if FORCE_CPU is set, use CPU. Preserves option to run on CPU.
"""
import os

# Set FORCE_CPU=1 or FORCE_CPU=true to use CPU even on Kaggle (e.g. if GPU doesn't work).
FORCE_CPU_ENV = "FORCE_CPU"


def _force_cpu() -> bool:
    v = os.environ.get(FORCE_CPU_ENV, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def is_kaggle() -> bool:
    """True if running inside a Kaggle kernel."""
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") != ""


def _use_gpu() -> bool:
    """Use GPU only on Kaggle when available and not forcing CPU."""
    if _force_cpu():
        return False
    if not is_kaggle():
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def use_cpu() -> bool:
    """True if we should run on CPU (local, or Kaggle with FORCE_CPU, or no GPU on Kaggle)."""
    return not _use_gpu()


def get_device_map() -> str:
    """
    device_map for HuggingFace from_pretrained.
    - "auto" only when on Kaggle with GPU and FORCE_CPU not set.
    - "cpu" otherwise (local, or FORCE_CPU, or no GPU on Kaggle).
    """
    return "cpu" if use_cpu() else "auto"


def print_device_info():
    """Log current device choice (call once at startup)."""
    if use_cpu():
        msg = "Using CPU"
        if is_kaggle() and _force_cpu():
            msg += " (FORCE_CPU set)"
    else:
        import torch
        msg = f"Using GPU: {torch.cuda.get_device_name(0)} [Kaggle]"
    print(msg, flush=True)
