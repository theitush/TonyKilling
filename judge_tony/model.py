"""Model architecture and loading utilities"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, BitsAndBytesConfig, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import ModelOutput
from peft import LoraConfig, get_peft_model
from typing import Tuple, Optional
from dataclasses import dataclass

from .config import TrainConfig
from .constants import (
    LORA_TARGET_MODULES,
    LORA_DROPOUT,
    LORA_BIAS,
    LORA_TASK_TYPE,
    QUANTIZATION_BITS,
    QUANTIZATION_USE_DOUBLE_QUANT,
    QUANTIZATION_TYPE,
    EPOCH_BRANCH_FORMAT,
    ADAPTER_CONFIG_FILE,
    MODEL_SAFETENSORS_FILE,
    MODEL_PYTORCH_FILE,
)


@dataclass
class RegressionOutput(ModelOutput):
    """Output class for regression models"""
    loss: Optional[torch.FloatTensor] = None
    predictions: torch.FloatTensor = None


def create_quantization_config(quantization_bits: int = QUANTIZATION_BITS) -> BitsAndBytesConfig:
    """
    Create quantization configuration for LoRA training

    Args:
        quantization_bits: Number of bits for quantization (4 or 8)

    Returns:
        BitsAndBytesConfig instance
    """
    if quantization_bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=QUANTIZATION_USE_DOUBLE_QUANT,
            bnb_4bit_quant_type=QUANTIZATION_TYPE,
        )
    elif quantization_bits == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
        )
    else:
        raise ValueError(f"quantization_bits must be 4 or 8, got {quantization_bits}")


class RegressionModelConfig(PretrainedConfig):
    """Configuration class for RegressionModel"""

    model_type = "regression_model"

    def __init__(
        self,
        base_model_name: str = "answerdotai/ModernBERT-base",
        model_type_detected: str = "encoder",
        hidden_size: int = 768,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        quantization_bits: int = 4,
        **kwargs
    ):
        """
        Args:
            base_model_name: HuggingFace model name to use as backbone
            model_type_detected: "encoder" or "decoder" based on architecture
            hidden_size: Hidden dimension of the backbone model
            use_lora: Whether LoRA was used during training
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            quantization_bits: Bits for quantization (4 or 8)
            **kwargs: Additional arguments passed to PretrainedConfig
        """
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.model_type_detected = model_type_detected
        self.hidden_size = hidden_size
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.quantization_bits = quantization_bits


class RegressionModel(PreTrainedModel):
    """Wrapper model for comedy score regression"""

    config_class = RegressionModelConfig

    def __init__(self, config: RegressionModelConfig, backbone: nn.Module = None):
        """
        Args:
            config: RegressionModelConfig with model settings
            backbone: Pre-trained transformer model (optional, for compatibility)
        """
        super().__init__(config)
        self.config = config

        # Backbone must be provided - either for training or when loading from checkpoint
        if backbone is None:
            raise ValueError(
                "backbone cannot be None. When creating a model, pass a backbone. "
                "When loading from checkpoint, use the proper loading method in run.py"
            )

        self.backbone = backbone

        self.head = nn.Linear(config.hidden_size, 1)
        self.model_type_str = config.model_type_detected

        if config.model_type_detected not in ["encoder", "decoder"]:
            raise ValueError(f"model_type must be 'encoder' or 'decoder', got {config.model_type_detected}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        """
        Forward pass

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Ground truth scores [batch_size] (optional)

        Returns:
            RegressionOutput with 'predictions' and optionally 'loss'
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Extract hidden state based on model type
        if self.model_type_str == "encoder":
            # Use [CLS] token (first token)
            hidden = outputs.last_hidden_state[:, 0]
        else:
            # Use last token
            hidden = outputs.last_hidden_state[:, -1]

        # Regression head - outputs unbounded values
        predictions = self.head(hidden).squeeze(-1)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(predictions, labels)

        return RegressionOutput(loss=loss, predictions=predictions)


def detect_model_type(model_name: str) -> str:
    """
    Detect if model is encoder or decoder based on name

    Simple heuristic: if "bert" is in the name (case-insensitive), it's an encoder

    Args:
        model_name: HuggingFace model name

    Returns:
        "encoder" or "decoder"
    """
    return "encoder" if "bert" in model_name.lower() else "decoder"


def load_model(config: TrainConfig) -> Tuple[RegressionModel, str]:
    """
    Load model from HuggingFace with optional LoRA and quantization

    Args:
        config: Training configuration

    Returns:
        Tuple of (RegressionModel, model_type)
    """
    model_type = detect_model_type(config.model_name)
    print(f"Detected model type: {model_type}")

    # Load backbone model config to get hidden size
    backbone_config = AutoConfig.from_pretrained(config.model_name)
    hidden_size = backbone_config.hidden_size

    # Create RegressionModelConfig
    regression_config = RegressionModelConfig(
        base_model_name=config.model_name,
        model_type_detected=model_type,
        hidden_size=hidden_size,
        use_lora=config.use_lora,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        quantization_bits=config.quantization_bits,
    )

    # Prepare quantization config if using LoRA
    quantization_config = None
    if config.use_lora:
        quantization_config = create_quantization_config(config.quantization_bits)
        print(f"Using {config.quantization_bits}-bit quantization with LoRA")

    # Load backbone
    # Note: Using device_map=None to let Trainer handle device placement
    # This works for both single-GPU and multi-GPU training (DataParallel/DDP)
    # device_map="auto" only works for inference or single-device scenarios
    backbone = AutoModel.from_pretrained(
        config.model_name,
        quantization_config=quantization_config,
        device_map=None,
        trust_remote_code=True,
    )

    # Enable gradient checkpointing for memory efficiency
    if hasattr(backbone, 'gradient_checkpointing_enable'):
        backbone.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing for memory efficiency")

    # Apply LoRA if enabled
    if config.use_lora:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias=LORA_BIAS,
            task_type=LORA_TASK_TYPE,
        )
        backbone = get_peft_model(backbone, lora_config)
        print(f"Applied LoRA with r={config.lora_r}, alpha={config.lora_alpha}")
        backbone.print_trainable_parameters()

    # Wrap in regression model with config and backbone
    model = RegressionModel(regression_config, backbone=backbone)

    return model, model_type


def validate_checkpoint(repo_id: str, branch_name: str, use_lora: bool) -> None:
    """
    Validate that checkpoint has required files

    Args:
        repo_id: HuggingFace repository ID
        branch_name: Branch name (e.g., "epoch-5")
        use_lora: Whether LoRA is expected

    Raises:
        ValueError: If checkpoint is invalid
    """
    from huggingface_hub import file_exists, HfApi
    from huggingface_hub.utils import HfHubHTTPError

    try:
        # Check if branch exists
        api = HfApi()
        refs = api.list_repo_refs(repo_id, repo_type="model")
        branch_names = [b.name for b in refs.branches]

        if branch_name not in branch_names:
            raise ValueError(
                f"Branch '{branch_name}' not found in repository '{repo_id}'. "
                f"Available branches: {branch_names}"
            )

        # Check for required files
        if use_lora:
            # For LoRA checkpoints, adapter files are REQUIRED
            has_adapter = file_exists(
                repo_id=repo_id,
                filename=ADAPTER_CONFIG_FILE,
                revision=branch_name,
            )
            if not has_adapter:
                raise ValueError(
                    f"Checkpoint at {repo_id} (branch: {branch_name}) is missing LoRA adapter files. "
                    f"This checkpoint was likely saved with an old version. "
                    f"Please retrain or use a checkpoint with adapter files."
                )

        # Check for model weights
        has_safetensors = file_exists(
            repo_id=repo_id,
            filename=MODEL_SAFETENSORS_FILE,
            revision=branch_name,
        )
        has_pytorch = file_exists(
            repo_id=repo_id,
            filename=MODEL_PYTORCH_FILE,
            revision=branch_name,
        )

        if not has_safetensors and not has_pytorch:
            raise ValueError(
                f"Checkpoint at {repo_id} (branch: {branch_name}) is missing model weights. "
                f"Expected either '{MODEL_SAFETENSORS_FILE}' or '{MODEL_PYTORCH_FILE}'."
            )

    except HfHubHTTPError as e:
        raise ValueError(
            f"Could not access repository '{repo_id}': {e}. "
            f"Make sure the repository exists and you have access."
        )


def load_checkpoint_model(repo_id: str, starting_epoch: int) -> Tuple[RegressionModel, str]:
    """
    Load a RegressionModel from a checkpoint repository

    This function handles loading a model from a HuggingFace Hub checkpoint,
    including LoRA adapters if present.

    Args:
        repo_id: HuggingFace repository ID
        starting_epoch: Epoch number to load (e.g., 5 for "epoch-5")

    Returns:
        Tuple of (loaded model, model_type)

    Raises:
        ValueError: If checkpoint is invalid or incompatible
    """
    from peft import PeftModel
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError
    from safetensors.torch import load_file

    branch_name = EPOCH_BRANCH_FORMAT.format(epoch=starting_epoch)
    print(f"Loading model from {repo_id} (branch: {branch_name})")

    # Load the RegressionModelConfig from the checkpoint
    try:
        regression_config = RegressionModelConfig.from_pretrained(
            repo_id,
            revision=branch_name,
            trust_remote_code=True,
        )
    except Exception as e:
        raise ValueError(
            f"Could not load model config from {repo_id} (branch: {branch_name}): {e}"
        )

    # Validate checkpoint structure
    print("Validating checkpoint structure...")
    validate_checkpoint(repo_id, branch_name, regression_config.use_lora)
    print("✓ Checkpoint validation passed")

    # Load base model with quantization if needed
    quantization_config = None
    if regression_config.use_lora:
        quantization_config = create_quantization_config(regression_config.quantization_bits)
        print(f"Using {regression_config.quantization_bits}-bit quantization")

    print(f"Loading base model: {regression_config.base_model_name}")
    try:
        backbone = AutoModel.from_pretrained(
            regression_config.base_model_name,
            quantization_config=quantization_config,
            device_map=None,  # Let Trainer handle device placement
            trust_remote_code=True,
        )
    except Exception as e:
        raise ValueError(
            f"Could not load base model '{regression_config.base_model_name}': {e}"
        )

    # Enable gradient checkpointing for memory efficiency
    if hasattr(backbone, 'gradient_checkpointing_enable'):
        backbone.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing for memory efficiency")

    # Load LoRA adapters if checkpoint used LoRA
    if regression_config.use_lora:
        print("Loading LoRA adapters from checkpoint...")
        try:
            backbone = PeftModel.from_pretrained(
                backbone,
                repo_id,
                revision=branch_name,
            )
            print("✓ Loaded LoRA adapters from checkpoint")
        except Exception as e:
            raise ValueError(
                f"Could not load LoRA adapters from {repo_id} (branch: {branch_name}): {e}. "
                f"This may indicate a corrupted checkpoint or version mismatch."
            )

    # Create RegressionModel with the loaded backbone
    model = RegressionModel(regression_config, backbone=backbone)

    # Load head weights from checkpoint
    print("Loading head weights from checkpoint...")
    try:
        # Try safetensors first (preferred format)
        try:
            weights_file = hf_hub_download(
                repo_id=repo_id,
                filename=MODEL_SAFETENSORS_FILE,
                revision=branch_name,
            )
            state_dict = load_file(weights_file)
            print(f"Loaded weights from {MODEL_SAFETENSORS_FILE}")
        except:
            # Fall back to pytorch_model.bin
            weights_file = hf_hub_download(
                repo_id=repo_id,
                filename=MODEL_PYTORCH_FILE,
                revision=branch_name,
            )
            state_dict = torch.load(weights_file, map_location='cpu')
            print(f"Loaded weights from {MODEL_PYTORCH_FILE}")

        # Extract and load head weights
        head_weights = {
            k.replace('head.', ''): v
            for k, v in state_dict.items()
            if k.startswith('head.')
        }

        if not head_weights:
            raise ValueError("No head weights found in checkpoint. Checkpoint may be corrupted.")

        model.head.load_state_dict(head_weights, strict=True)
        print(f"✓ Loaded head weights ({len(head_weights)} parameters)")

    except Exception as e:
        raise ValueError(
            f"Could not load model weights from {repo_id} (branch: {branch_name}): {e}"
        )

    # Sanity check: verify model can do a forward pass
    print("Running sanity check on loaded model...")
    try:
        dummy_input_ids = torch.randint(0, 1000, (1, 10))
        dummy_attention_mask = torch.ones(1, 10)

        with torch.no_grad():
            output = model(
                input_ids=dummy_input_ids,
                attention_mask=dummy_attention_mask,
            )

        if output.predictions is None or torch.isnan(output.predictions).any() or torch.isinf(output.predictions).any():
            raise ValueError("Model produced invalid outputs (NaN or Inf)")

        print("✓ Model sanity check passed")

    except Exception as e:
        raise ValueError(f"Model sanity check failed: {e}. The loaded checkpoint may be corrupted.")

    print(f"✓ Successfully loaded checkpoint from epoch {starting_epoch}")
    return model, regression_config.model_type_detected


# Register the model with AutoModel for easy loading
try:
    from transformers import AutoModel as HFAutoModel
    HFAutoModel.register(RegressionModelConfig, RegressionModel)
    print("✓ Registered RegressionModel with AutoModel")
except Exception as e:
    # Registration might fail in some contexts, that's okay
    print(f"Note: Could not register with AutoModel ({e}). Use RegressionModel.from_pretrained() instead.")
