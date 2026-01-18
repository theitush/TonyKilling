"""Main entrypoint for training Judge Tony models"""

import pandas as pd
from transformers import AutoTokenizer
from typing import Tuple, Dict

from .config import TrainConfig
from .model import load_model
from .data import JudgeTonyDataset
from .trainer import JudgeTonyTrainer, create_training_args, EpochCheckpointCallback
from .evaluate import predict, compute_metrics


def train(
    config: TrainConfig,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_to_drive: bool = True,
    drive_path: str = "/content/drive/MyDrive/judge_tony_checkpoints",
) -> Tuple[object, object, Dict[str, float]]:
    """
    Main training function

    Args:
        config: Training configuration
        train_df: Training DataFrame with 'transcript' and 'score' columns
        test_df: Test DataFrame with 'transcript' and 'score' columns
        save_to_drive: Whether to save checkpoints to Google Drive (for Colab)
        drive_path: Path in Google Drive to save checkpoints

    Returns:
        Tuple of (model, tokenizer, eval_results)
        - model: Trained RegressionModel
        - tokenizer: HuggingFace tokenizer
        - eval_results: Dict with test set metrics (mse, mae)
    """
    print("=" * 50)
    print("Judge Tony Training Pipeline")
    print("=" * 50)
    print(f"Model: {config.model_name}")
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    print(f"Max length: {config.max_length}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.lr}")
    print(f"Epochs: {config.epochs}")
    print(f"LoRA: {config.use_lora}")
    print("=" * 50)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)

    # Add padding token if missing (common for decoder models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = JudgeTonyDataset(train_df, tokenizer, config.max_length)
    test_dataset = JudgeTonyDataset(test_df, tokenizer, config.max_length)

    # Load model
    print("\nLoading model...")
    model, model_type = load_model(config)
    print(f"Model type: {model_type}")

    # Create training arguments
    training_args = create_training_args(config)

    # Create checkpoint callback
    checkpoint_callback = EpochCheckpointCallback(
        checkpoint_dir=config.output_dir,
        save_to_drive=save_to_drive,
        drive_path=drive_path,
    )

    # Create trainer
    print("\nInitializing trainer...")
    trainer = JudgeTonyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        callbacks=[checkpoint_callback],
    )

    # Train
    print("\nStarting training...")
    print("=" * 50)
    trainer.train()

    # Evaluate on test set
    print("\n" + "=" * 50)
    print("Evaluating on test set...")
    test_texts = test_df["transcript"].tolist()
    test_labels = test_df["score"].values

    predictions = predict(
        model,
        tokenizer,
        test_texts,
        batch_size=config.batch_size,
        max_length=config.max_length,
    )

    eval_results = compute_metrics(predictions, test_labels)

    print("\nTest Set Results:")
    print(f"  MSE: {eval_results['mse']:.4f}")
    print(f"  MAE: {eval_results['mae']:.4f}")
    print("=" * 50)

    # Save model and tokenizer
    print(f"\nSaving model to {config.output_dir}...")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    print("\nTraining complete!")

    return model, tokenizer, eval_results
