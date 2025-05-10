import torch
from torch.utils.data import Dataset
import pandas as pd
import json
import os
from typing import List, Dict, Optional, Union
from datasets import load_dataset
from transformers import PreTrainedTokenizer


class MambaTextDataset(Dataset):
    """
    Dataset for fine-tuning Mamba models on text data.
    
    Handles tokenization and formatting for both causal language modeling
    and sequence classification tasks.
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[Union[int, float]]] = None,
        tokenizer: PreTrainedTokenizer = None,
        max_length: int = 512,
        task_type: str = "causal_lm",
        padding: str = "max_length",
        truncation: bool = True,
    ):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text examples
            labels: List of labels for sequence classification
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            task_type: Type of task ("causal_lm" or "classification")
            padding: Padding strategy for tokenization
            truncation: Whether to truncate sequences longer than max_length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        self.padding = padding
        self.truncation = truncation
        
        if task_type == "classification" and labels is None:
            raise ValueError("Labels must be provided for classification tasks")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt"
        )
        
        # Convert to tensors and squeeze batch dimension added by tokenizer
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        if self.task_type == "causal_lm":
            # For causal language modeling, labels are the input ids
            # Shifted right for next token prediction
            labels = input_ids.clone()
            
            # Return dictionary with tensors
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        else:
            # For classification tasks
            label = self.labels[idx]
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": torch.tensor(label)
            }


class PairwiseTextDataset(Dataset):
    """
    Dataset for fine-tuning Mamba models on pairwise text data
    suitable for reward-based learning.
    
    This dataset format is particularly useful for the DeltaMamba training
    approach where reward signals guide parameter updates.
    """
    
    def __init__(
        self,
        preferred_texts: List[str],
        rejected_texts: List[str],
        tokenizer: PreTrainedTokenizer = None,
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True,
        reward_for_preferred: float = 1.0,
        reward_for_rejected: float = -1.0,
    ):
        """
        Initialize the pairwise dataset.
        
        Args:
            preferred_texts: List of preferred text examples
            rejected_texts: List of rejected/less preferred text examples
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            padding: Padding strategy for tokenization
            truncation: Whether to truncate sequences longer than max_length
            reward_for_preferred: Reward value for preferred examples
            reward_for_rejected: Reward value for rejected examples
        """
        assert len(preferred_texts) == len(rejected_texts), "Lists must have the same length"
        
        self.preferred_texts = preferred_texts
        self.rejected_texts = rejected_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.reward_for_preferred = reward_for_preferred
        self.reward_for_rejected = reward_for_rejected
    
    def __len__(self):
        # Returns the combined length of both preferred and rejected examples
        return len(self.preferred_texts) + len(self.rejected_texts)
    
    def __getitem__(self, idx):
        # Determine if this is a preferred or rejected example
        is_preferred = idx < len(self.preferred_texts)
        
        if is_preferred:
            text = self.preferred_texts[idx]
            reward = self.reward_for_preferred
        else:
            # Adjust index for rejected examples
            adj_idx = idx - len(self.preferred_texts)
            text = self.rejected_texts[adj_idx]
            reward = self.reward_for_rejected
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt"
        )
        
        # Convert to tensors and squeeze batch dimension
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # For causal language modeling with reward signals
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "reward": torch.tensor(reward, dtype=torch.float)
        }


def load_squad_dataset(tokenizer: PreTrainedTokenizer, max_length: int = 384, version: str = "v2"):
    """
    Load and prepare SQuAD dataset for question answering task.
    
    Args:
        tokenizer: Tokenizer for text processing
        max_length: Maximum sequence length
        version: SQuAD version ("v1" or "v2")
    
    Returns:
        Tuple of train and validation datasets
    """
    # Load SQuAD from HuggingFace datasets
    squad_version = "squad" if version == "v1" else "squad_v2"
    squad = load_dataset(squad_version)
    
    train_texts = []
    train_labels = []
    
    # Process training data
    for example in squad["train"]:
        context = example["context"]
        question = example["question"]
        
        # Format as: "Question: {question} Context: {context}"
        formatted_text = f"Question: {question} Context: {context}"
        
        # For simplicity, we'll use the answer as a classification target
        # In a real implementation, you might want to handle this differently
        if example["answers"]["text"]:
            # Use the first answer if available
            answer = example["answers"]["text"][0]
            train_texts.append(formatted_text)
            train_labels.append(answer)
    
    # Repeat for validation data
    val_texts = []
    val_labels = []
    
    for example in squad["validation"]:
        context = example["context"]
        question = example["question"]
        formatted_text = f"Question: {question} Context: {context}"
        
        if example["answers"]["text"]:
            answer = example["answers"]["text"][0]
            val_texts.append(formatted_text)
            val_labels.append(answer)
    
    # Create datasets
    train_dataset = MambaTextDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=max_length,
        task_type="causal_lm"  # We'll phrase this as language modeling
    )
    
    val_dataset = MambaTextDataset(
        texts=val_texts,
        labels=val_labels,
        tokenizer=tokenizer,
        max_length=max_length,
        task_type="causal_lm"
    )
    
    return train_dataset, val_dataset


def load_mnli_dataset(tokenizer: PreTrainedTokenizer, max_length: int = 256):
    """
    Load and prepare MultiNLI dataset for natural language inference task.
    
    Args:
        tokenizer: Tokenizer for text processing
        max_length: Maximum sequence length
    
    Returns:
        Tuple of train and validation datasets
    """
    # Load MNLI from HuggingFace datasets
    mnli = load_dataset("glue", "mnli")
    
    # Process training data
    train_texts = []
    train_labels = []
    
    for example in mnli["train"]:
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        
        # Format as: "Premise: {premise} Hypothesis: {hypothesis} Label:"
        formatted_text = f"Premise: {premise}