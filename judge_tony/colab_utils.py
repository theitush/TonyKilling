"""Utilities for running Judge Tony training on Google Colab"""

import torch
import gc
import os
import json
from pathlib import Path
from typing import Optional, Dict


def clear_gpu_memory():
    """Clear GPU memory cache - run this before training"""
    print("Clearing GPU memory...")

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Force garbage collection
    gc.collect()

    # Print memory stats
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    else:
        print("CUDA not available")


def check_gpu():
    """Check GPU availability and specs"""
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available! Training will be very slow on CPU.")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"GPU: {gpu_name}")
    print(f"Total Memory: {gpu_memory:.2f}GB")

    return True


def setup_colab_training():
    """
    Setup function to run at the start of your Colab notebook

    Returns:
        bool: True if setup successful
    """
    print("=" * 50)
    print("Colab Training Setup")
    print("=" * 50)

    # Check GPU
    if not check_gpu():
        return False

    # Clear memory
    clear_gpu_memory()

    # Set environment variables for better memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    print("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

    print("=" * 50)
    print("Setup complete! Ready to train.")
    print("=" * 50)

    return True


def save_checkpoint(
    model,
    eval_results: Dict,
    epoch: int,
    checkpoint_dir: str,
):
    """
    Save model checkpoint and eval results locally

    Args:
        model: The model to save
        eval_results: Dictionary with evaluation metrics
        epoch: Current epoch number
        checkpoint_dir: Local checkpoint directory
    """
    from peft import PeftModel

    # Create local checkpoint dir
    local_ckpt_path = f"{checkpoint_dir}/epoch_{epoch}"
    os.makedirs(local_ckpt_path, exist_ok=True)

    # Save model
    print(f"\nSaving checkpoint for epoch {epoch}...")

    # Check if the backbone is a PEFT model and save adapters separately
    if isinstance(model.backbone, PeftModel):
        print("  Saving LoRA adapter weights...")
        model.backbone.save_pretrained(local_ckpt_path)

    # Save the full model (RegressionModel config + all weights)
    model.save_pretrained(local_ckpt_path)

    # Save eval results
    results_path = f"{local_ckpt_path}/eval_results.json"
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)

    print(f"✓ Saved checkpoint to {local_ckpt_path}")

    return local_ckpt_path


def load_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find and return path to the latest checkpoint

    Args:
        checkpoint_dir: Local checkpoint directory

    Returns:
        Path to latest checkpoint directory, or None if no checkpoints found
    """
    if os.path.exists(checkpoint_dir):
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("epoch_")]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split("_")[1]))
            path = os.path.join(checkpoint_dir, latest)
            print(f"Found local checkpoint: {path}")
            return path

    print("No checkpoints found")
    return None


def cleanup_old_epoch_checkpoints(checkpoint_dir: str, keep_last_n: int):
    """
    Delete old epoch checkpoints, keeping only the N most recent.

    This helps manage disk space during training by removing old checkpoints
    after they've been uploaded to HuggingFace Hub.

    Args:
        checkpoint_dir: Local checkpoint directory
        keep_last_n: Number of most recent checkpoints to keep
    """
    import shutil

    if not os.path.exists(checkpoint_dir):
        return

    # Find all epoch_* directories
    epoch_dirs = [d for d in os.listdir(checkpoint_dir) if d.startswith("epoch_")]

    if len(epoch_dirs) <= keep_last_n:
        # Nothing to clean up
        return

    # Sort by epoch number
    epoch_dirs.sort(key=lambda x: int(x.split("_")[1]))

    # Delete all except the last N
    dirs_to_delete = epoch_dirs[:-keep_last_n]

    for dir_name in dirs_to_delete:
        dir_path = os.path.join(checkpoint_dir, dir_name)
        print(f"🗑️  Deleting old checkpoint: {dir_name}")
        shutil.rmtree(dir_path)

    if dirs_to_delete:
        print(f"✓ Cleaned up {len(dirs_to_delete)} old checkpoint(s), kept {keep_last_n} most recent")
