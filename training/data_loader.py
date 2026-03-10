"""Data loading and preprocessing for VLM compliance finetuning.

Loads the synthetic dataset from JSONL, splits into train/val by example,
flattens records into individual question-answer pairs, formats them as
chat conversations in Mistral's multimodal format, and provides a PyTorch
Dataset with a custom collator for batching images and text together.

Usage:
    from data_loader import (
        load_records, train_val_split, flatten, format_chat,
        ComplianceDataset, make_collate_fn,
    )

    records = load_records("data/train/dataset.jsonl")
    train_records, val_records = train_val_split(records)
    train_examples = flatten(train_records, "data/train")
    dataset = ComplianceDataset(train_examples)
    collate_fn = make_collate_fn(processor)
"""

from __future__ import annotations

import json
import os
import random
import sys
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor

# Add parent directory to path so we can import evaluate.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluate import SYSTEM_PROMPT, prompt_builder


def load_records(path: str) -> list[dict[str, Any]]:
    """Load all records from a dataset JSONL file.

    Each line is one JSON object representing a single example with
    an image, specification text, questions, and metadata.

    Args:
        path: Path to the dataset JSONL file.

    Returns:
        List of record dicts, one per example.
    """
    with open(path) as f:
        return [json.loads(line) for line in f]


def train_val_split(
    records: list[dict[str, Any]],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records into train and validation sets by example.

    Splits by example rather than by question so that no image
    appears in both sets. Creates a shallow copy to avoid mutating
    the input list.

    Args:
        records: List of dataset records.
        val_ratio: Fraction of records to use for validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_records, val_records).
    """
    records = list(records)
    random.seed(seed)
    random.shuffle(records)
    train_n = int((1.0 - val_ratio) * len(records))
    return records[:train_n], records[train_n:]


def flatten(
    records: list[dict[str, Any]],
    dataset_dir: str,
) -> list[dict[str, Any]]:
    """Flatten records into one dict per question.

    Each record contains ~27 questions sharing the same image and
    specification. This explodes them into individual training
    examples, each with a resolved image path and all fields needed
    to build a chat conversation.

    Args:
        records: List of dataset records from load_records.
        dataset_dir: Directory containing the dataset, used to resolve
            relative image paths (e.g. "data/train").

    Returns:
        List of flat example dicts, one per question.
    """
    examples: list[dict[str, Any]] = []
    for record in records:
        for q in record["questions"]:
            example = {
                "image_path": os.path.join(dataset_dir, record["image"]),
                "spec_text": record["spec_text"],
                "question_type": q["type"],
                "question": q["question"],
                "answer": q["answer"],
                "reasoning": q["reasoning"],
                "annotation_level": record["metadata"]["annotation_level"],
                "rule_complexity": record["metadata"]["rule_complexity"],
            }
            examples.append(example)
    return examples


def oversample_no(
    examples: list[dict[str, Any]],
    target_ratio: float = 0.3,
) -> list[dict[str, Any]]:
    """Oversample 'No' compliance examples to reduce Yes bias.

    The dataset is ~90% Yes / 10% No for compliance questions.
    Training on this imbalance teaches the model to always say Yes.
    This function duplicates No examples until they reach target_ratio
    of compliance examples. Non-compliance questions (measurements,
    audit, etc.) are passed through unchanged.

    Args:
        examples: Flat examples from flatten().
        target_ratio: Desired fraction of No among compliance
            examples. 0.3 means 30% No, 70% Yes.

    Returns:
        New list with oversampled No examples added.
    """
    compliance_yes: list[dict[str, Any]] = []
    compliance_no: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []

    for ex in examples:
        if ex["question_type"] == "per_component_compliance":
            if ex["answer"] == "No":
                compliance_no.append(ex)
            else:
                compliance_yes.append(ex)
        else:
            other.append(ex)

    n_yes = len(compliance_yes)
    n_no = len(compliance_no)
    print(
        f"Before oversampling: {n_yes} Yes, {n_no} No "
        f"({n_no / (n_yes + n_no):.1%} No)"
    )

    # n_no_target = target_ratio * n_yes / (1 - target_ratio)
    n_no_target = int(target_ratio * n_yes / (1 - target_ratio))

    if n_no_target <= n_no:
        print("Already at or above target ratio, no oversampling needed")
        return examples

    extra = random.choices(compliance_no, k=n_no_target - n_no)

    n_total_no = n_no + len(extra)
    print(
        f"After oversampling: {n_yes} Yes, {n_total_no} No "
        f"({n_total_no / (n_yes + n_total_no):.1%} No)"
    )

    return compliance_yes + compliance_no + extra + other


def format_chat(example: dict[str, Any]) -> dict[str, Any]:
    """Format a flat example as a multimodal chat conversation.

    Produces the Mistral/OpenAI-style messages format with three turns:
    system prompt, user turn (image + text prompt), and assistant turn
    (answer + reasoning). The user prompt is built by prompt_builder
    from evaluate.py so that training and evaluation prompts match.

    The image is referenced via a file:// URL in the user message,
    which the training framework resolves when loading the batch.

    Args:
        example: A flat example dict from flatten.

    Returns:
        Dict with keys:
            messages: List of system/user/assistant message dicts.
            image_path: Path to the image file.
    """
    prompt_text = prompt_builder(
        example["question_type"],
        example["spec_text"],
        example["question"],
    )

    assistant_content = f"{example['answer']}. {example['reasoning']}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"file://{example['image_path']}"},
                },
                {"type": "text", "text": prompt_text},
            ],
        },
        {"role": "assistant", "content": assistant_content},
    ]

    return {
        "messages": messages,
        "image_path": example["image_path"],
    }


class ComplianceDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapping flat examples with on-the-fly formatting.

    Each item returns a dict with chat messages and a loaded PIL image.
    Chat formatting is deferred to __getitem__ to avoid storing all
    formatted conversations in memory.

    Args:
        examples: List of flat example dicts from flatten.
    """

    def __init__(self, examples: list[dict[str, Any]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one training example with messages and image.

        Args:
            idx: Index into the examples list.

        Returns:
            Dict with 'messages' (chat format) and 'image' (PIL Image).
        """
        ex = self.examples[idx]
        chat = format_chat(ex)
        image = Image.open(ex["image_path"]).convert("RGB")
        return {
            "messages": chat["messages"],
            "image": image,
        }


def make_collate_fn(
    processor: AutoProcessor,
    max_seq_len: int = 4096,
):
    """Create a collate function for batching images and chat text.

    The collator applies the chat template, tokenizes text, processes
    images, and creates labels with padding and image tokens masked
    to -100 so the loss ignores them.

    Args:
        processor: The model's processor (tokenizer + image processor).
        max_seq_len: Maximum sequence length for truncation.

    Returns:
        Collate function compatible with PyTorch DataLoader.
    """
    pad_token_id = processor.tokenizer.pad_token_id
    image_token_id = getattr(processor, "image_token_id", None)

    def collate_fn(examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        image_list: list[Image.Image] = []
        message_list: list[list[dict[str, Any]]] = []
        for example in examples:
            image_list.append(example["image"])
            message_list.append(example["messages"])

        chat_texts = processor.apply_chat_template(
            message_list,
            add_generation_prompt=False,
            tokenize=False,
        )

        batch = processor(
            text=chat_texts,
            images=image_list,
            padding="longest",
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()

        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        if image_token_id is not None and image_token_id != pad_token_id:
            labels[labels == image_token_id] = -100

        batch["labels"] = labels
        return batch

    return collate_fn
