"""Model architecture and loading utilities"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from typing import Tuple

from .config import TrainConfig


class RegressionModel(nn.Module):
    """Wrapper model for comedy score regression"""

    def __init__(self, backbone: nn.Module, hidden_size: int, model_type: str):
        """
        Args:
            backbone: Pre-trained transformer model
            hidden_size: Hidden dimension of the backbone
            model_type: "encoder" or "decoder"
        """
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(hidden_size, 1)
        self.model_type = model_type

        if model_type not in ["encoder", "decoder"]:
            raise ValueError(f"model_type must be 'encoder' or 'decoder', got {model_type}")

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
            dict with 'logits' (predictions) and optionally 'loss'
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Extract hidden state based on model type
        if self.model_type == "encoder":
            # Use [CLS] token (first token)
            hidden = outputs.last_hidden_state[:, 0]
        else:
            # Use last token
            hidden = outputs.last_hidden_state[:, -1]

        # Regression head with sigmoid activation
        logits = torch.sigmoid(self.head(hidden)).squeeze(-1)

        output = {"logits": logits}

        # Compute loss if labels provided
        if labels is not None:
            loss_fct = nn.MSELoss()
            output["loss"] = loss_fct(logits, labels)

        return type('Output', (), output)()  # Create object with attributes


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

    # Load model config
    model_config = AutoConfig.from_pretrained(config.model_name)
    hidden_size = model_config.hidden_size

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
    backbone = AutoModel.from_pretrained(
        config.model_name,
        quantization_config=quantization_config,
        device_map="auto" if config.use_lora else None,
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

    # Wrap in regression model
    model = RegressionModel(backbone, hidden_size, model_type)

    # If using fp16, convert the regression head to float16
    if config.fp16 and not config.use_lora:
        model.head = model.head.half()
        print("Converted regression head to float16 for fp16 training")

    return model, model_type
