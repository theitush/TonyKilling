"""Constants for Judge Tony training pipeline"""

# Checkpoint versioning
EPOCH_BRANCH_FORMAT = "epoch-{epoch}"

# LoRA configuration
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_DROPOUT = 0.05
LORA_BIAS = "none"
LORA_TASK_TYPE = "FEATURE_EXTRACTION"

# Quantization
QUANTIZATION_BITS = 4
QUANTIZATION_COMPUTE_DTYPE = "float16"  # Will be converted to torch.float16
QUANTIZATION_USE_DOUBLE_QUANT = True
QUANTIZATION_TYPE = "nf4"

# Checkpoint files
ADAPTER_CONFIG_FILE = "adapter_config.json"
ADAPTER_MODEL_FILE = "adapter_model.bin"
MODEL_SAFETENSORS_FILE = "model.safetensors"
MODEL_PYTORCH_FILE = "pytorch_model.bin"
CONFIG_FILE = "config.json"
TRAINING_CONFIG_FILE = "training_config.json"
EVAL_RESULTS_FILE = "eval_results.json"
README_FILE = "README.md"
