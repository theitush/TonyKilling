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
    """Callback to save checkpoints and eval results at the end of each epoch"""

    def __init__(
        self,
        checkpoint_dir: str,
        save_to_drive: bool = True,
        drive_path: str = "/content/drive/MyDrive/judge_tony_checkpoints"
    ):
        """
        Args:
            checkpoint_dir: Local directory to save checkpoints
            save_to_drive: Whether to copy to Google Drive
            drive_path: Path in Google Drive for backups
        """
        self.checkpoint_dir = checkpoint_dir
        self.save_to_drive = save_to_drive
        self.drive_path = drive_path

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of each epoch"""
        from .colab_utils import save_checkpoint

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

        # Save checkpoint
        save_checkpoint(
            model=model,
            eval_results=eval_results,
            epoch=epoch,
            checkpoint_dir=self.checkpoint_dir,
            save_to_drive=self.save_to_drive,
            drive_path=self.drive_path
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
