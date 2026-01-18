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


def mount_google_drive(mount_path: str = "/content/drive") -> bool:
    """
    Mount Google Drive for persistent checkpoint storage

    Args:
        mount_path: Where to mount Google Drive (default: /content/drive)

    Returns:
        bool: True if successful
    """
    try:
        from google.colab import drive
        drive.mount(mount_path)
        print(f"✓ Google Drive mounted at {mount_path}")
        return True
    except ImportError:
        print("Warning: Not running in Google Colab, skipping Drive mount")
        return False
    except Exception as e:
        print(f"Error mounting Google Drive: {e}")
        return False


def save_checkpoint(
    model,
    eval_results: Dict,
    epoch: int,
    checkpoint_dir: str,
    save_to_drive: bool = True,
    drive_path: str = "/content/drive/MyDrive/judge_tony_checkpoints"
):
    """
    Save model checkpoint and eval results

    Args:
        model: The model to save
        eval_results: Dictionary with evaluation metrics
        epoch: Current epoch number
        checkpoint_dir: Local checkpoint directory
        save_to_drive: Whether to also copy to Google Drive (for Colab)
        drive_path: Path in Google Drive to save checkpoints
    """
    # Create local checkpoint dir
    local_ckpt_path = f"{checkpoint_dir}/epoch_{epoch}"
    os.makedirs(local_ckpt_path, exist_ok=True)

    # Save model
    print(f"\nSaving checkpoint for epoch {epoch}...")
    model.save_pretrained(local_ckpt_path)

    # Save eval results
    results_path = f"{local_ckpt_path}/eval_results.json"
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)

    print(f"✓ Saved to {local_ckpt_path}")

    # Copy to Google Drive if requested
    if save_to_drive:
        try:
            drive_ckpt_path = f"{drive_path}/epoch_{epoch}"
            os.makedirs(drive_ckpt_path, exist_ok=True)

            # Copy model files
            import shutil
            shutil.copytree(local_ckpt_path, drive_ckpt_path, dirs_exist_ok=True)

            print(f"✓ Backed up to Google Drive: {drive_ckpt_path}")
        except Exception as e:
            print(f"Warning: Could not save to Google Drive: {e}")
            print("  (Make sure Google Drive is mounted)")


def load_latest_checkpoint(
    checkpoint_dir: str,
    drive_path: Optional[str] = "/content/drive/MyDrive/judge_tony_checkpoints"
) -> Optional[str]:
    """
    Find and return path to the latest checkpoint

    Args:
        checkpoint_dir: Local checkpoint directory
        drive_path: Path in Google Drive (will check here first if it exists)

    Returns:
        Path to latest checkpoint directory, or None if no checkpoints found
    """
    # First try Google Drive if path exists
    if drive_path and os.path.exists(drive_path):
        checkpoints = [d for d in os.listdir(drive_path) if d.startswith("epoch_")]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split("_")[1]))
            path = os.path.join(drive_path, latest)
            print(f"Found checkpoint in Google Drive: {path}")
            return path

    # Fall back to local directory
    if os.path.exists(checkpoint_dir):
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("epoch_")]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split("_")[1]))
            path = os.path.join(checkpoint_dir, latest)
            print(f"Found local checkpoint: {path}")
            return path

    print("No checkpoints found")
    return None
