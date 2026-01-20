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
    from .hub_utils import setup_hf_auth, get_repo_name, get_latest_epoch_branch, extract_base_model_from_repo

    print("=" * 50)
    print("Judge Tony Training Pipeline")
    print("=" * 50)

    # Check for auto-resume
    starting_epoch = 0
    resume_checkpoint_path = None
    base_model_for_card = config.model_name

    if config.resume_from_checkpoint:
        print(f"\n🔄 Resume mode detected: {config.resume_from_checkpoint}")

        # Get latest epoch from HuggingFace Hub
        latest_epoch_info = get_latest_epoch_branch(config.resume_from_checkpoint)

        if latest_epoch_info:
            branch_name, epoch_num = latest_epoch_info
            starting_epoch = epoch_num
            resume_checkpoint_path = config.resume_from_checkpoint
            print(f"✓ Found checkpoint at {branch_name} (epoch {epoch_num})")
            print(f"📥 Will resume from epoch {epoch_num}, continuing to epoch {config.epochs}")

            # Extract base model for model card generation
            base_model_extracted = extract_base_model_from_repo(config.resume_from_checkpoint)
            if base_model_extracted:
                base_model_for_card = base_model_extracted
                print(f"✓ Base model: {base_model_for_card}")
        else:
            print(f"⚠️  No checkpoints found in {config.resume_from_checkpoint}")
            print("Starting new training instead...")
            config.resume_from_checkpoint = None

    print(f"\nModel: {config.model_name}")
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    print(f"Max length: {config.max_length}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.lr}")
    print(f"Epochs: {config.epochs}")
    print(f"LoRA: {config.use_lora}")
    print(f"Upload to Hub: {config.upload_to_hub}")
    if starting_epoch > 0:
        print(f"Starting epoch: {starting_epoch + 1}")
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

    # Load tokenizer (from checkpoint if resuming, otherwise from base model)
    print("\nLoading tokenizer...")
    tokenizer_source = config.model_name
    if config.resume_from_checkpoint and starting_epoch > 0:
        # Load tokenizer from checkpoint branch
        from .model import RegressionModel
        tokenizer_source = config.resume_from_checkpoint
        print(f"Loading tokenizer from checkpoint: {tokenizer_source}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)

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
    if config.resume_from_checkpoint and starting_epoch > 0:
        # Resume from checkpoint
        from .model import RegressionModel
        branch_name = f"epoch-{starting_epoch}"
        print(f"Loading model from {config.resume_from_checkpoint} (branch: {branch_name})")
        model = RegressionModel.from_pretrained(
            config.resume_from_checkpoint,
            revision=branch_name,
            trust_remote_code=True,
        )
        print(f"✓ Loaded checkpoint from epoch {starting_epoch}")
        model_type = model.config.model_type_detected
    else:
        # Load fresh model
        model, model_type = load_model(config)

    print(f"Model type: {model_type}")

    # Create training arguments
    training_args = create_training_args(config)

    # Create checkpoint callback
    checkpoint_callback = EpochCheckpointCallback(
        checkpoint_dir=config.output_dir,
        hf_repo_name=hf_repo_name,
        base_model_name=base_model_for_card,
        upload_to_hub=config.upload_to_hub,
        keep_last_n_epochs=config.keep_last_n_epochs,
        starting_epoch=starting_epoch,
        train_config=config,
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
