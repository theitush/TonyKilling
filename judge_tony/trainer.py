"""Custom trainer for Judge Tony fine-tuning"""

import torch
from transformers import Trainer, TrainingArguments, TrainerCallback
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class JudgeTonyTrainer(Trainer):
    """Custom trainer with MSE loss for regression"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store loss function for logging
        self.loss_fct = torch.nn.MSELoss()

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

        # Get predictions (unbounded regression values)
        predictions = outputs.predictions

        # Ensure labels match predictions dtype for fp16 training
        labels = labels.to(predictions.dtype)

        # Compute MSE loss
        loss = self.loss_fct(predictions, labels)

        return (loss, outputs) if return_outputs else loss


class EpochCheckpointCallback(TrainerCallback):
    """Callback to save checkpoints and upload to HuggingFace Hub after evaluation"""

    def __init__(
        self,
        checkpoint_dir: str,
        hf_repo_name: Optional[str] = None,
        base_model_name: Optional[str] = None,
        upload_to_hub: bool = True,
        trainer: Optional['JudgeTonyTrainer'] = None,
        keep_last_n_epochs: int = 1,
        starting_epoch: int = 0,
        train_config: Optional[object] = None,
    ):
        """
        Args:
            checkpoint_dir: Local directory to save checkpoints
            hf_repo_name: HuggingFace repository name (e.g., "username/judge-tony-qwen")
            base_model_name: Base model name for model card generation
            upload_to_hub: Whether to upload to HuggingFace Hub
            trainer: Reference to the trainer instance (for accessing loss function)
            keep_last_n_epochs: Number of most recent epoch checkpoints to keep locally
            starting_epoch: Epoch number to start from (for resumed training)
            train_config: TrainConfig instance to save with checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        self.hf_repo_name = hf_repo_name
        self.base_model_name = base_model_name
        self.upload_to_hub = upload_to_hub
        self.trainer = trainer
        self.keep_last_n_epochs = keep_last_n_epochs
        self.starting_epoch = starting_epoch
        self.train_config = train_config

    def _create_qq_plots(self, train_preds: np.ndarray, train_actuals: np.ndarray,
                        val_preds: np.ndarray, val_actuals: np.ndarray, epoch: int):
        """
        Create side-by-side QQ plots comparing predictions vs actual scores

        Args:
            train_preds: Array of training predicted scores
            train_actuals: Array of training actual scores
            val_preds: Array of validation predicted scores
            val_actuals: Array of validation actual scores
            epoch: Current epoch number
        """
        # normalize predictions to score range [-0.5, 1.5] for better visualization
        train_preds = np.array([max(-0.5, min(1.5, p)) for p in train_preds])
        val_preds = np.array([max(-0.5, min(1.5, p)) for p in val_preds])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Training set plot
        sns.scatterplot(x=train_preds, y=train_actuals, alpha=0.5, ax=ax1)
        ax1.plot([0, 1], [0, 1], 'r--', label='Perfect prediction')
        ax1.set_xlabel('Predicted Score')
        ax1.set_ylabel('Actual Score')
        ax1.set_title(f'Epoch {epoch} - Training Set: Predictions vs Actual Scores')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Validation set plot
        sns.scatterplot(x=val_preds, y=val_actuals, alpha=0.5, ax=ax2)
        ax2.plot([0, 1], [0, 1], 'r--', label='Perfect prediction')
        ax2.set_xlabel('Predicted Score')
        ax2.set_ylabel('Actual Score')
        ax2.set_title(f'Epoch {epoch} - Validation Set: Predictions vs Actual Scores')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _get_predictions(self, trainer, dataset):
        """
        Get predictions for a dataset

        Args:
            trainer: The Trainer instance
            dataset: Dataset to get predictions for

        Returns:
            Tuple of (predictions, actuals)
        """
        predictions_output = trainer.predict(dataset)
        predictions = predictions_output.predictions.squeeze()
        actuals = predictions_output.label_ids.squeeze()
        return predictions, actuals

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        """Called after evaluation completes - this ensures we have current epoch's metrics"""
        from .colab_utils import save_checkpoint
        from .hub_utils import upload_checkpoint_to_hub, save_training_config

        # Only save/upload at epoch boundaries (when eval_strategy="epoch")
        # state.epoch is the current epoch (e.g., 1.0, 2.0, 3.0)
        if not state.epoch.is_integer():
            return

        # Calculate actual epoch number (accounting for resumed training)
        epoch = int(state.epoch) + self.starting_epoch

        # Get loss function from stored trainer reference
        loss_function = str(self.trainer.loss_fct) if self.trainer and hasattr(self.trainer, 'loss_fct') else None

        # Generate QQ plots for train and validation sets
        if self.trainer:
            print(f"\n📊 Generating QQ plots for epoch {epoch}...")

            # Get train and validation dataset predictions
            train_preds, train_actuals = None, None
            val_preds, val_actuals = None, None

            if hasattr(self.trainer, 'train_dataset') and self.trainer.train_dataset is not None:
                train_preds, train_actuals = self._get_predictions(self.trainer, self.trainer.train_dataset)

            if hasattr(self.trainer, 'eval_dataset') and self.trainer.eval_dataset is not None:
                val_preds, val_actuals = self._get_predictions(self.trainer, self.trainer.eval_dataset)

            # Create side-by-side plots if both datasets are available
            if train_preds is not None and val_preds is not None:
                self._create_qq_plots(train_preds, train_actuals, val_preds, val_actuals, epoch)
            else:
                print("⚠️ Both train and validation datasets needed for QQ plots.")
        else:
            print("⚠️ Trainer instance not available; skipping QQ plots.")

        # Get train_loss from state.log_history (most recent entry with 'loss' key for current epoch)
        train_loss = None
        for log_entry in reversed(state.log_history):
            if 'loss' in log_entry and log_entry.get('epoch', 0) <= epoch:
                train_loss = log_entry['loss']
                break

        # Build eval results from the metrics passed to this callback
        eval_results = {
            'epoch': epoch,
            'loss_function': loss_function,
            'train_loss': train_loss,
            'eval_loss': metrics.get('eval_loss'),
            'eval_runtime': metrics.get('eval_runtime'),
            'eval_samples_per_second': metrics.get('eval_samples_per_second'),
        }

        # Save checkpoint locally
        checkpoint_path = save_checkpoint(
            model=model,
            eval_results=eval_results,
            epoch=epoch,
            checkpoint_dir=self.checkpoint_dir,
        )

        # Save training config to checkpoint
        if self.train_config is not None:
            save_training_config(self.train_config, checkpoint_path)

        # Upload to HuggingFace Hub
        if self.upload_to_hub and self.hf_repo_name and self.base_model_name:
            upload_checkpoint_to_hub(
                checkpoint_dir=checkpoint_path,
                repo_name=self.hf_repo_name,
                epoch=epoch,
                eval_results=eval_results,
                base_model_name=self.base_model_name,
            )

        # Clean up old checkpoints to save disk space
        from .colab_utils import cleanup_old_epoch_checkpoints
        cleanup_old_epoch_checkpoints(self.checkpoint_dir, self.keep_last_n_epochs)


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
        # Performance
        fp16=config.fp16,
        dataloader_num_workers=0,
        # Reporting
        report_to=["tensorboard"],
        disable_tqdm=False,
        # Misc
        remove_unused_columns=False,  # Keep all columns for custom forward
    )
