"""Main entrypoint for training Judge Tony models"""

import pandas as pd
from transformers import AutoTokenizer, AutoModel
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
    base_model_for_card = config.model_name

    if config.resume_from_checkpoint:
        print(f"\n🔄 Resume mode detected: {config.resume_from_checkpoint}")

        # Get latest epoch from HuggingFace Hub
        latest_epoch_info = get_latest_epoch_branch(config.resume_from_checkpoint)

        if latest_epoch_info:
            branch_name, epoch_num = latest_epoch_info
            starting_epoch = epoch_num
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

    # Load tokenizer (always from base model, even when resuming)
    print("\nLoading tokenizer...")
    tokenizer_source = base_model_for_card if (config.resume_from_checkpoint and starting_epoch > 0) else config.model_name
    if config.resume_from_checkpoint and starting_epoch > 0:
        print(f"Loading tokenizer from base model: {tokenizer_source}")

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
        # Resume from checkpoint - use special loading for PEFT models
        from .model import RegressionModel, RegressionModelConfig
        from peft import PeftModel
        import torch

        branch_name = f"epoch-{starting_epoch}"
        print(f"Loading model from {config.resume_from_checkpoint} (branch: {branch_name})")

        # First, load the RegressionModelConfig from the checkpoint
        regression_config = RegressionModelConfig.from_pretrained(
            config.resume_from_checkpoint,
            revision=branch_name,
            trust_remote_code=True,
        )

        # Now load the base model with quantization
        quantization_config = None
        if regression_config.use_lora:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        print(f"Loading base model: {regression_config.base_model_name}")
        backbone = AutoModel.from_pretrained(
            regression_config.base_model_name,
            quantization_config=quantization_config,
            device_map=None,
            trust_remote_code=True,
        )

        # Try to load PEFT adapters from checkpoint (new format)
        # If adapter files don't exist (old format), we'll load the full weights instead
        adapter_loaded = False
        if regression_config.use_lora:
            try:
                print("Attempting to load LoRA adapters from checkpoint...")
                from huggingface_hub import file_exists

                # Check if adapter files exist
                has_adapter_config = file_exists(
                    repo_id=config.resume_from_checkpoint,
                    filename="adapter_config.json",
                    revision=branch_name,
                )

                if has_adapter_config:
                    backbone = PeftModel.from_pretrained(
                        backbone,
                        config.resume_from_checkpoint,
                        revision=branch_name,
                    )
                    adapter_loaded = True
                    print("✓ Loaded LoRA adapters from checkpoint")
                else:
                    print("⚠️  No adapter files found (old checkpoint format)")
                    print("   Will load full merged weights instead...")
            except Exception as e:
                print(f"⚠️  Could not load LoRA adapters: {e}")
                print("   Will load full merged weights instead...")

        # Create RegressionModel with the loaded backbone
        model = RegressionModel(regression_config, backbone=backbone)

        # Load weights from checkpoint
        import os
        from huggingface_hub import hf_hub_download

        # Download model.safetensors or pytorch_model.bin
        try:
            # Try safetensors first
            weights_file = hf_hub_download(
                repo_id=config.resume_from_checkpoint,
                filename="model.safetensors",
                revision=branch_name,
            )
            from safetensors.torch import load_file
            state_dict = load_file(weights_file)
        except:
            # Fall back to pytorch_model.bin
            weights_file = hf_hub_download(
                repo_id=config.resume_from_checkpoint,
                filename="pytorch_model.bin",
                revision=branch_name,
            )
            state_dict = torch.load(weights_file, map_location='cpu')

        if adapter_loaded:
            # New format: only load head weights
            print("Loading head weights from checkpoint...")
            head_weights = {k.replace('head.', ''): v for k, v in state_dict.items() if k.startswith('head.')}
            model.head.load_state_dict(head_weights, strict=False)
        else:
            # Old format: load full model weights (backbone + head)
            print("Loading full model weights from checkpoint...")
            # Filter out keys that don't match (e.g., due to architecture changes)
            model.load_state_dict(state_dict, strict=False)

        print(f"✓ Loaded checkpoint from epoch {starting_epoch}")
        model_type = regression_config.model_type_detected
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
