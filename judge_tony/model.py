"""Model architecture and loading utilities"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, BitsAndBytesConfig, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import ModelOutput
from peft import LoraConfig, get_peft_model
from typing import Tuple, Optional
from dataclasses import dataclass

from .config import TrainConfig


@dataclass
class RegressionOutput(ModelOutput):
    """Output class for regression models"""
    loss: Optional[torch.FloatTensor] = None
    predictions: torch.FloatTensor = None


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

    def _load_backbone_from_config(self, config: RegressionModelConfig) -> nn.Module:
        """
        Load the backbone model from the configuration.
        Used when resuming from checkpoint.

        Args:
            config: RegressionModelConfig with model settings

        Returns:
            Loaded backbone model
        """
        # Prepare quantization config if using LoRA
        quantization_config = None
        if config.use_lora:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        # Load backbone
        backbone = AutoModel.from_pretrained(
            config.base_model_name,
            quantization_config=quantization_config,
            device_map=None,
            trust_remote_code=True,
        )

        # Enable gradient checkpointing for memory efficiency
        if hasattr(backbone, 'gradient_checkpointing_enable'):
            backbone.gradient_checkpointing_enable()

        # Apply LoRA if enabled
        if config.use_lora:
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            backbone = get_peft_model(backbone, lora_config)

        return backbone

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
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
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
            target_modules=["q_proj", "v_proj"],  # Common attention modules
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        backbone = get_peft_model(backbone, lora_config)
        print(f"Applied LoRA with r={config.lora_r}, alpha={config.lora_alpha}")
        backbone.print_trainable_parameters()

    # Wrap in regression model with config and backbone
    model = RegressionModel(regression_config, backbone=backbone)

    return model, model_type


# Register the model with AutoModel for easy loading
try:
    from transformers import AutoModel as HFAutoModel
    HFAutoModel.register(RegressionModelConfig, RegressionModel)
    print("✓ Registered RegressionModel with AutoModel")
except Exception as e:
    # Registration might fail in some contexts, that's okay
    print(f"Note: Could not register with AutoModel ({e}). Use RegressionModel.from_pretrained() instead.")
