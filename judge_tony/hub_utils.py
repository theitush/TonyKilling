"""HuggingFace Hub integration utilities"""

import os
import json
from typing import Dict, Optional
from pathlib import Path

from huggingface_hub import HfApi, login, whoami, create_repo, upload_folder, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

from .constants import EPOCH_BRANCH_FORMAT


def setup_hf_auth(token: Optional[str] = None) -> Optional[str]:
    """
    Setup HuggingFace authentication

    Args:
        token: HF token (if None, will try to use HF_TOKEN env var or existing login)

    Returns:
        HuggingFace username if authenticated, None otherwise
    """
    # If token explicitly provided (not from env var), use it to login
    if token is not None:
        try:
            login(token=token, add_to_git_credential=False)
            print("✓ Logged in to HuggingFace Hub with provided token")
        except Exception as e:
            print(f"Warning: Could not login with provided token: {e}")
            return None

    # Try to get current user (will use HF_TOKEN env var automatically if set)
    try:
        user_info = whoami()
        username = user_info.get("name")
        print(f"✓ Authenticated as HuggingFace user: {username}")
        return username
    except Exception as e:
        print(f"Warning: Not authenticated to HuggingFace Hub: {e}")
        print("To authenticate:")
        print("1. Get a token from https://huggingface.co/settings/tokens")
        print("2. Set environment variable: HF_TOKEN=your_token")
        print("3. Or run: huggingface-cli login")
        return None


def get_repo_name(base_model_name: str, hf_username: str, repo_prefix: str = "judge-tony") -> str:
    """
    Generate repository name based on base model

    Args:
        base_model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-0.5B")
        hf_username: HuggingFace username
        repo_prefix: Prefix for repo name

    Returns:
        Full repo name (e.g., "username/judge-tony-qwen2.5-0.5b")
    """
    # Extract model name from path and use it directly
    # E.g., "Qwen/Qwen2.5-0.5B" -> "qwen2.5-0.5b"
    # E.g., "meta-llama/Llama-2-7b" -> "llama-2-7b"
    model_name = base_model_name.split("/")[-1].lower()

    repo_name = f"{hf_username}/{repo_prefix}-{model_name}"
    return repo_name


def extract_base_model_from_repo(repo_name: str, repo_prefix: str = "judge-tony") -> Optional[str]:
    """
    Extract base model name from checkpoint repository name

    Args:
        repo_name: Checkpoint repo name (e.g., "itacas/judge-tony-qwen3-4b")
        repo_prefix: Prefix used in repo names

    Returns:
        Base model name if found (e.g., "Qwen/Qwen3-4B"), None otherwise
    """
    from .config import MODEL_CONFIGS

    # Extract the model part from repo name
    # "itacas/judge-tony-qwen3-4b" -> "qwen3-4b"
    if "/" in repo_name:
        repo_name = repo_name.split("/")[-1]

    if not repo_name.startswith(repo_prefix):
        return None

    # Remove prefix: "judge-tony-qwen3-4b" -> "qwen3-4b"
    model_part = repo_name[len(repo_prefix):].lstrip("-")

    # Try to match against MODEL_CONFIGS keys
    for base_model_name in MODEL_CONFIGS.keys():
        # Extract model name from base model path
        # "Qwen/Qwen3-4B" -> "qwen3-4b"
        base_model_short = base_model_name.split("/")[-1].lower()

        if base_model_short == model_part:
            return base_model_name

    return None


def get_latest_epoch_branch(repo_id: str) -> Optional[tuple]:
    """
    Query HuggingFace Hub to find the latest epoch branch

    Args:
        repo_id: Repository ID (e.g., "itacas/judge-tony-qwen3-4b")

    Returns:
        Tuple of (branch_name, epoch_number) if found, None otherwise
        Example: ("epoch-5", 5)
    """
    try:
        api = HfApi()
        refs = api.list_repo_refs(repo_id, repo_type="model")

        # Find all epoch branches (using EPOCH_BRANCH_FORMAT pattern)
        epoch_branches = []
        for branch in refs.branches:
            if branch.name.startswith("epoch-"):
                try:
                    # Extract epoch number from branch name
                    epoch_num = int(branch.name.split("-")[1])
                    epoch_branches.append((branch.name, epoch_num))
                except (ValueError, IndexError):
                    continue

        if not epoch_branches:
            return None

        # Return the branch with highest epoch number
        return max(epoch_branches, key=lambda x: x[1])

    except Exception as e:
        print(f"Could not query repository {repo_id}: {e}")
        return None


def save_training_config(config, checkpoint_dir: str):
    """
    Save TrainConfig to checkpoint directory as JSON

    Args:
        config: TrainConfig instance
        checkpoint_dir: Directory to save config to
    """
    from dataclasses import asdict

    config_path = os.path.join(checkpoint_dir, "training_config.json")

    # Convert config to dict
    config_dict = asdict(config)

    # Save as JSON
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"✓ Saved training config to {config_path}")


def create_model_card(
    repo_name: str,
    base_model_name: str,
    eval_results: Dict,
    epoch: int,
    training_args: Optional[Dict] = None,
) -> str:
    """
    Create a model card (README.md) with training info and metrics

    Args:
        repo_name: Repository name (e.g., "username/judge-tony-qwen")
        base_model_name: Base model used
        eval_results: Evaluation metrics dictionary
        epoch: Current epoch number
        training_args: Training hyperparameters (optional)

    Returns:
        Model card content as string
    """
    # Extract key metrics
    eval_loss = eval_results.get("eval_loss", "N/A")
    eval_runtime = eval_results.get("eval_runtime", "N/A")

    card = f"""---
license: apache-2.0
base_model: {base_model_name}
tags:
- regression
- comedy-rating
- fine-tuned
library_name: transformers
---

# {repo_name.split('/')[-1].title().replace('-', ' ')}

This model is a fine-tuned regression model based on **{base_model_name}** for predicting comedy scores.

## Model Description

- **Base Model:** {base_model_name}
- **Task:** Regression (comedy score prediction)
- **Architecture:** Custom regression head on top of transformer backbone
- **Training Epoch:** {epoch}

## Training Details

### Evaluation Metrics (Epoch {epoch})

- **Evaluation Loss:** {eval_loss}
- **Evaluation Runtime:** {eval_runtime}s

"""

    # Add training arguments if provided
    if training_args:
        card += "### Training Hyperparameters\n\n"
        card += "```json\n"
        card += json.dumps(training_args, indent=2)
        card += "\n```\n\n"

    card += """## Usage

```python
from transformers import AutoTokenizer, AutoModel

# Load model and tokenizer
model = AutoModel.from_pretrained("{repo_name}")
tokenizer = AutoTokenizer.from_pretrained("{repo_name}")

# Inference
inputs = tokenizer("Your text here", return_tensors="pt")
outputs = model(**inputs)
score = outputs.logits.item()
```

## Citation

If you use this model, please cite:

```bibtex
@misc{{judge-tony,
  author = {{itacas}},
  title = {{Judge Tony: Comedy Rating Model}},
  year = {{2025}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/{repo_name}}}
}}
```

---

*This model was trained using the Judge Tony training pipeline.*
""".format(repo_name=repo_name)

    return card


def upload_checkpoint_to_hub(
    checkpoint_dir: str,
    repo_name: str,
    epoch: int,
    eval_results: Dict,
    base_model_name: str,
    commit_message: Optional[str] = None,
) -> bool:
    """
    Upload checkpoint to HuggingFace Hub with branch-based versioning

    Each epoch is uploaded to its own branch (epoch-1, epoch-2, etc.) to preserve
    all checkpoint versions.

    Usage example:
        # Load specific epoch checkpoint
        model = AutoModel.from_pretrained("username/judge-tony-qwen", revision="epoch-3")

    Args:
        checkpoint_dir: Local directory containing checkpoint files
        repo_name: HuggingFace repo name (e.g., "username/judge-tony-qwen")
        epoch: Epoch number
        eval_results: Evaluation results dictionary
        base_model_name: Base model name
        commit_message: Custom commit message (optional)

    Returns:
        True if upload succeeded, False otherwise
    """
    try:
        api = HfApi()

        # Create repository if it doesn't exist
        try:
            create_repo(repo_name, exist_ok=True, repo_type="model")
            print(f"✓ Repository ready: https://huggingface.co/{repo_name}")
        except Exception as e:
            print(f"Note: Repository may already exist: {e}")

        # Generate model card
        model_card = create_model_card(
            repo_name=repo_name,
            base_model_name=base_model_name,
            eval_results=eval_results,
            epoch=epoch,
        )

        # Save model card to checkpoint directory
        readme_path = os.path.join(checkpoint_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(model_card)

        # Save eval results as JSON
        results_path = os.path.join(checkpoint_dir, "eval_results.json")
        with open(results_path, "w") as f:
            json.dump(eval_results, f, indent=2)

        # Determine revision (branch/tag) and commit message
        if commit_message is None:
            eval_loss = eval_results.get("eval_loss", "N/A")
            commit_message = f"Upload epoch {epoch} checkpoint (eval_loss: {eval_loss})"

        # Upload to epoch-specific branch to preserve all checkpoints
        epoch_branch = EPOCH_BRANCH_FORMAT.format(epoch=epoch)
        print(f"⬆️  Uploading checkpoint to {repo_name} (branch: {epoch_branch})...")

        # Create branch from main if it doesn't exist
        try:
            api.create_branch(
                repo_id=repo_name,
                branch=epoch_branch,
                repo_type="model",
                exist_ok=True,
            )
        except Exception as e:
            print(f"Note: Branch creation info: {e}")

        api.upload_folder(
            folder_path=checkpoint_dir,
            repo_id=repo_name,
            repo_type="model",
            revision=epoch_branch,
            commit_message=commit_message,
            create_pr=False,
            allow_patterns=["config.json", "model.safetensors", "generation_config.json", "README.md", "eval_results.json", "training_config.json"],
        )

        print(f"✓ Uploaded to https://huggingface.co/{repo_name}/tree/{epoch_branch}")

        return True

    except HfHubHTTPError as e:
        print(f"❌ HTTP error uploading to HuggingFace Hub: {e}")
        return False
    except Exception as e:
        print(f"❌ Error uploading to HuggingFace Hub: {e}")
        print("Training will continue, but checkpoint was not uploaded.")
        return False


def upload_existing_checkpoint(
    checkpoint_path: str,
    repo_name: str,
    base_model_name: str,
    epoch: Optional[int] = None,
) -> bool:
    """
    Upload an existing checkpoint directory to HuggingFace Hub

    Useful for retroactively uploading checkpoints or re-uploading

    Args:
        checkpoint_path: Path to checkpoint directory
        repo_name: HuggingFace repo name
        base_model_name: Base model name
        epoch: Epoch number (will try to infer from path if None)

    Returns:
        True if upload succeeded, False otherwise
    """
    # Try to infer epoch from path if not provided
    if epoch is None:
        path_name = Path(checkpoint_path).name
        if "epoch" in path_name:
            try:
                epoch = int(path_name.split("epoch")[-1].strip("_-"))
            except ValueError:
                epoch = 0
        else:
            epoch = 0

    # Load eval results if they exist
    eval_results = {}
    results_path = os.path.join(checkpoint_path, "eval_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            eval_results = json.load(f)

    return upload_checkpoint_to_hub(
        checkpoint_dir=checkpoint_path,
        repo_name=repo_name,
        epoch=epoch,
        eval_results=eval_results,
        base_model_name=base_model_name,
    )
