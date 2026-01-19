"""Configuration for Judge Tony fine-tuning pipeline"""

from dataclasses import dataclass
from typing import Optional


# Model configurations
MODELS = [
    # Best overall after fine-tuning, dominates benchmarks
    "Qwen/Qwen3-4B",

    # Smaller Qwen, still strong, very efficient
    "Qwen/Qwen3-1.7B",

    # Tiny Qwen, surprisingly capable for 0.6B
    "Qwen/Qwen3-0.6B",

    # Microsoft's reasoning beast, punches way above its weight
    "microsoft/Phi-4-mini-instruct",

    # Most tunable - gains the most from fine-tuning
    "meta-llama/Llama-3.2-1B-Instruct",

    # Good middle ground for Llama family
    "meta-llama/Llama-3.2-3B-Instruct",

    # HuggingFace's fully open model, beats Llama/Qwen at 3B
    "HuggingFaceTB/SmolLM3-3B",

    # Google's latest small model, good multilingual
    "google/gemma-3-1b-it",

    # ModernBERT - state-of-the-art encoder model
    "answerdotai/ModernBERT-base",
    "answerdotai/ModernBERT-large",
]


# Model-specific configurations
MODEL_CONFIGS = {
    # Qwen models
    "Qwen/Qwen3-4B": {
        "max_length": 1024,
        "batch_size": 4,
        "lr": 2e-5,
        "use_lora": True,
        "lora_r": 16,
    },
    "Qwen/Qwen3-1.7B": {
        "max_length": 1024,
        "batch_size": 8,
        "lr": 2e-5,
        "use_lora": False,
    },
    "Qwen/Qwen3-0.6B": {
        "max_length": 1024,
        "batch_size": 16,
        "lr": 3e-5,
        "use_lora": False,
    },

    # Microsoft Phi
    "microsoft/Phi-4-mini-instruct": {
        "max_length": 1024,
        "batch_size": 4,
        "lr": 2e-5,
        "use_lora": True,
        "lora_r": 16,
    },

    # Meta Llama
    "meta-llama/Llama-3.2-1B-Instruct": {
        "max_length": 1024,
        "batch_size": 8,
        "lr": 2e-5,
        "use_lora": False,
    },
    "meta-llama/Llama-3.2-3B-Instruct": {
        "max_length": 1024,
        "batch_size": 4,
        "lr": 2e-5,
        "use_lora": True,
        "lora_r": 16,
    },

    # HuggingFace SmolLM
    "HuggingFaceTB/SmolLM3-3B": {
        "max_length": 1024,
        "batch_size": 4,
        "lr": 2e-5,
        "use_lora": True,
        "lora_r": 16,
    },

    # Google Gemma
    "google/gemma-3-1b-it": {
        "max_length": 1024,
        "batch_size": 8,
        "lr": 2e-5,
        "use_lora": False,
    },

    # ModernBERT - encoder models (different approach than decoder-only)
    "answerdotai/ModernBERT-base": {
        "max_length": 512,  # Encoders typically use shorter context
        "batch_size": 16,
        "lr": 5e-5,  # BERT models often need higher LR
        "use_lora": False,
    },
    "answerdotai/ModernBERT-large": {
        "max_length": 512,
        "batch_size": 8,
        "lr": 3e-5,
        "use_lora": True,
        "lora_r": 16,
    },
}


@dataclass
class TrainConfig:
    """Configuration for training Judge Tony models"""

    # Model configuration
    model_name: str
    max_length: int = 1024

    # Training hyperparameters
    batch_size: int = 4  # Reduced for Colab compatibility
    lr: float = 2e-5
    epochs: int = 5

    # LoRA configuration
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    quantization_bits: int = 4  # 4-bit quantization when LoRA is enabled

    # Checkpointing and evaluation
    output_dir: str = "./judge_tony"
    save_strategy: str = "epoch"  # "epoch", "steps", or "no"
    eval_steps: int = 5  # Evaluate every N steps

    # HuggingFace Hub integration
    upload_to_hub: bool = True  # Whether to upload checkpoints to HuggingFace Hub
    hf_username: Optional[str] = None  # HuggingFace username (auto-detected if not set)
    hf_repo_prefix: str = "judge-tony"  # Prefix for repo names (e.g., "judge-tony-qwen")

    # Logging
    logging_steps: int = 1  # Log every step for visibility

    # Optional overrides
    warmup_steps: int = 0
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4  # Accumulate to effective batch of 16
    fp16: bool = True  # Mixed precision training for memory savings

    def __post_init__(self):
        """Validate configuration"""
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if self.max_length < 1:
            raise ValueError("max_length must be >= 1")
        if self.save_strategy not in ["epoch", "steps", "no"]:
            raise ValueError("save_strategy must be 'epoch', 'steps', or 'no'")

    @classmethod
    def from_model(cls, model_name: str, **overrides) -> "TrainConfig":
        """Create a TrainConfig from a model name using predefined configs.

        Args:
            model_name: Name of the model (must be in MODEL_CONFIGS)
            **overrides: Additional parameters to override defaults

        Returns:
            TrainConfig instance with model-specific settings

        Example:
            config = TrainConfig.from_model("Qwen/Qwen3-4B", epochs=10)
        """
        if model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"No predefined config for {model_name}. "
                f"Available models: {list(MODEL_CONFIGS.keys())}"
            )

        # Start with model-specific config
        config_dict = {"model_name": model_name, **MODEL_CONFIGS[model_name]}

        # Apply any overrides
        config_dict.update(overrides)

        return cls(**config_dict)
