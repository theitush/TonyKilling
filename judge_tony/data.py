"""Dataset class for Judge Tony fine-tuning"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Dict


class JudgeTonyDataset(Dataset):
    """PyTorch Dataset for comedy transcripts with scores"""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 1024):
        """
        Args:
            df: DataFrame with 'transcript' and 'score' columns
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Validate required columns
        if "transcript" not in df.columns:
            raise ValueError("DataFrame must have 'transcript' column")
        if "score" not in df.columns:
            raise ValueError("DataFrame must have 'score' column")

        print(f"Loaded dataset with {len(self.df)} examples")
        print(f"Score range: [{self.df['score'].min():.3f}, {self.df['score'].max():.3f}]")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example

        Returns:
            Dict with keys: input_ids, attention_mask, labels
        """
        row = self.df.iloc[idx]

        # Tokenize transcript
        encoding = self.tokenizer(
            row["transcript"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Extract tensors and squeeze batch dimension
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Get score as float tensor
        score = torch.tensor(row["score"], dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": score,
        }
