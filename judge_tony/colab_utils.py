"""Utilities for running Judge Tony training on Google Colab"""

import torch
import gc
import os


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
