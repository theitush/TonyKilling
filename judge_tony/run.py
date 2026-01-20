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
) -> Tuple[object, object, Dict[str, float]]:
    """
    Main training function

    Args:
        config: Training configuration
        train_df: Training DataFrame with 'transcript' and 'score' columns
        test_df: Test DataFrame with 'transcript' and 'score' columns

    Returns:
        Tuple of (model, tokenizer, eval_results)
        - model: Trained RegressionModel
        - tokenizer: HuggingFace tokenizer
        - eval_results: Dict with test set metrics (mse, mae)
    """
    from .hub_utils import setup_hf_auth, get_repo_name

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
    print(f"Upload to Hub: {config.upload_to_hub}")
    print("=" * 50)

    # Setup HuggingFace authentication
    hf_username = None
    hf_repo_name = None
    if config.upload_to_hub:
        print("\nSetting up HuggingFace Hub authentication...")
        hf_username = setup_hf_auth()

        # Use configured username or auto-detected one
        if config.hf_username:
            hf_username = config.hf_username

        if hf_username:
            hf_repo_name = get_repo_name(
                base_model_name=config.model_name,
                hf_username=hf_username,
                repo_prefix=config.hf_repo_prefix,
            )
            print(f"Will upload to: {hf_repo_name}")
        else:
            print("Warning: HuggingFace authentication failed. Checkpoints will only be saved locally.")
            config.upload_to_hub = False
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
        hf_repo_name=hf_repo_name,
        base_model_name=config.model_name,
        upload_to_hub=config.upload_to_hub,
        keep_last_n_epochs=config.keep_last_n_epochs,
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

    # Set trainer reference in callback for accessing loss function
    checkpoint_callback.trainer = trainer

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
    print("\nTraining complete!")

    return model, tokenizer, eval_results
