"""Custom trainer for Judge Tony fine-tuning"""

import torch
from transformers import Trainer, TrainingArguments, TrainerCallback
from typing import Dict, Optional
import numpy as np


class JudgeTonyTrainer(Trainer):
    """Custom trainer with MSE loss for regression"""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom loss computation using MSE

        Args:
            model: The model being trained
            inputs: Dict with input_ids, attention_mask, labels
            return_outputs: Whether to return model outputs

        Returns:
            Loss tensor, or (loss, outputs) if return_outputs=True
        """
        labels = inputs.pop("labels")

        # Forward pass
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        # Get predictions (already passed through sigmoid in model)
        logits = outputs.logits

        # Ensure labels match logits dtype for fp16 training
        labels = labels.to(logits.dtype)

        # Compute MSE loss
        loss_fct = torch.nn.MSELoss()
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


class EpochCheckpointCallback(TrainerCallback):
    """Callback to save checkpoints and upload to HuggingFace Hub at the end of each epoch"""

    def __init__(
        self,
        checkpoint_dir: str,
        hf_repo_name: Optional[str] = None,
        base_model_name: Optional[str] = None,
        upload_to_hub: bool = True,
    ):
        """
        Args:
            checkpoint_dir: Local directory to save checkpoints
            hf_repo_name: HuggingFace repository name (e.g., "username/judge-tony-qwen")
            base_model_name: Base model name for model card generation
            upload_to_hub: Whether to upload to HuggingFace Hub
        """
        self.checkpoint_dir = checkpoint_dir
        self.hf_repo_name = hf_repo_name
        self.base_model_name = base_model_name
        self.upload_to_hub = upload_to_hub
        self.best_eval_loss = float('inf')
        self.best_epoch = None

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of each epoch"""
        from .colab_utils import save_checkpoint
        from .hub_utils import upload_checkpoint_to_hub

        # Get current epoch (state.epoch is 1-indexed during training)
        epoch = int(state.epoch)

        # Get eval metrics from the latest log
        eval_results = {}
        if state.log_history:
            # Find the most recent eval metrics
            for log in reversed(state.log_history):
                if 'eval_loss' in log:
                    eval_results = {
                        'epoch': epoch,
                        'eval_loss': log.get('eval_loss'),
                        'eval_runtime': log.get('eval_runtime'),
                        'eval_samples_per_second': log.get('eval_samples_per_second'),
                    }
                    break

        # Save checkpoint locally
        checkpoint_path = save_checkpoint(
            model=model,
            eval_results=eval_results,
            epoch=epoch,
            checkpoint_dir=self.checkpoint_dir,
        )

        # Track best checkpoint
        current_loss = eval_results.get('eval_loss', float('inf'))
        is_best = current_loss < self.best_eval_loss
        if is_best:
            self.best_eval_loss = current_loss
            self.best_epoch = epoch
            print(f"🏆 New best checkpoint! Loss: {current_loss:.4f}")

        # Upload to HuggingFace Hub
        if self.upload_to_hub and self.hf_repo_name and self.base_model_name:
            upload_checkpoint_to_hub(
                checkpoint_dir=checkpoint_path,
                repo_name=self.hf_repo_name,
                epoch=epoch,
                eval_results=eval_results,
                base_model_name=self.base_model_name,
                is_best=is_best,
            )


def create_training_args(config) -> TrainingArguments:
    """
    Create TrainingArguments from config

    Args:
        config: TrainConfig object

    Returns:
        TrainingArguments for HuggingFace Trainer
    """
    return TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        # Logging
        logging_dir=f"{config.output_dir}/logs",
        logging_steps=config.logging_steps,
        logging_strategy="steps",
        # Evaluation
        eval_strategy=config.save_strategy,
        eval_steps=config.eval_steps if config.save_strategy == "steps" else None,
        # Saving
        save_strategy=config.save_strategy,
        save_steps=config.eval_steps if config.save_strategy == "steps" else None,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        # Performance
        fp16=config.fp16,
        dataloader_num_workers=0,
        # Reporting
        report_to=["tensorboard"],
        disable_tqdm=False,
        # Misc
        remove_unused_columns=False,  # Keep all columns for custom forward
    )
