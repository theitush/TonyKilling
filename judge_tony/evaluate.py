"""Evaluation and inference utilities"""

import torch
import numpy as np
from typing import List, Dict
from tqdm import tqdm


def predict(model, tokenizer, texts: List[str], batch_size: int = 8, max_length: int = 1024) -> np.ndarray:
    """
    Batch inference on a list of texts

    Args:
        model: Trained RegressionModel
        tokenizer: HuggingFace tokenizer
        texts: List of transcript strings
        batch_size: Batch size for inference
        max_length: Maximum sequence length

    Returns:
        Numpy array of predicted scores (0-1)
    """
    model.eval()
    device = next(model.parameters()).device
    predictions = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
            batch_texts = texts[i : i + batch_size]

            # Tokenize batch
            encoding = tokenizer(
                batch_texts,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # Move to device
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_preds = outputs.logits.cpu().numpy()

            predictions.extend(batch_preds)

    return np.array(predictions)


def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics

    Args:
        predictions: Predicted scores
        labels: Ground truth scores

    Returns:
        Dict with mse and mae
    """
    mse = np.mean((predictions - labels) ** 2)
    mae = np.mean(np.abs(predictions - labels))

    return {
        "mse": float(mse),
        "mae": float(mae),
    }
