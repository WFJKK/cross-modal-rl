"""Data loader for VLM engineering compliance finetuning.

Loads the dataset JSONL, splits into train/val at the record level,
flattens records into individual training examples (one per question),
and formats them as chat messages for Mistral finetuning.

Includes oversampling of minority class ("No" compliance answers) to
balance the training set.

Usage::

    from data_loader import (
        load_and_split, flatten_records, format_chat_mistral,
        oversample_minority,
    )

    train_records, val_records = load_and_split("dataset/dataset.jsonl")
    train_examples = flatten_records(train_records, "dataset/")
    train_examples = oversample_minority(train_examples)
    train_chat = [format_chat_mistral(ex) for ex in train_examples]
"""

from __future__ import annotations

import json
import os
import random
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evaluate import SYSTEM_PROMPT, prompt_builder


def load_and_split(
    dataset_path: str,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load dataset JSONL and split into train/val at the record level.

    Splits by record (not by question) so all questions from one plate
    stay together in the same split. This prevents data leakage where
    the model sees the same image in both train and val.

    Args:
        dataset_path: Path to dataset.jsonl.
        val_ratio: Fraction of records to hold out for validation.
        seed: Random seed for reproducible shuffling.

    Returns:
        Tuple of (train_records, val_records).
    """
    with open(dataset_path, "r") as f:
        records = [json.loads(line) for line in f]

    rng = random.Random(seed)
    rng.shuffle(records)

    split_idx = int(len(records) * (1 - val_ratio))
    train_records = records[:split_idx]
    val_records = records[split_idx:]

    return train_records, val_records


def flatten_records(
    records: list[dict[str, Any]],
    dataset_dir: str,
    question_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Flatten records into individual training examples.

    Each record contains ~23 questions sharing one image and spec.
    This function explodes them into separate examples, one per question,
    each with the image path, spec text, and metadata attached.

    Args:
        records: List of dataset records from load_and_split.
        dataset_dir: Root directory of the dataset (for resolving image paths).
        question_types: Optional filter — only include these question types.
            If None, all question types are included.

    Returns:
        List of flat example dicts with keys: image_path, spec_text,
        question, question_type, answer, reasoning, annotation_level,
        rule_complexity.
    """
    examples: list[dict[str, Any]] = []

    for record in records:
        image_path = os.path.join(dataset_dir, record["image"])
        spec_text = record["spec_text"]
        metadata = record["metadata"]

        for q in record["questions"]:
            if question_types and q["type"] not in question_types:
                continue

            examples.append(
                {
                    "image_path": image_path,
                    "spec_text": spec_text,
                    "question": q["question"],
                    "question_type": q["type"],
                    "answer": q["answer"],
                    "reasoning": q["reasoning"],
                    "annotation_level": metadata["annotation_level"],
                    "rule_complexity": metadata["rule_complexity"],
                }
            )

    return examples


def oversample_minority(
    examples: list[dict[str, Any]],
    target_ratio: float = 0.5,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Oversample 'No' compliance examples to balance the training set.

    The dataset is ~90 % Yes / 10 % No for compliance questions. Training
    on this imbalance teaches the model to always say Yes. This function
    duplicates No examples until the Yes/No ratio reaches target_ratio.

    Non-compliance questions are passed through unchanged.

    Args:
        examples: List of flat examples from flatten_records.
        target_ratio: Desired fraction of No among compliance examples.
            0.5 means equal Yes and No.
        seed: Random seed for reproducible sampling.

    Returns:
        New list with oversampled No examples added. Original examples
        are not modified.
    """
    rng = random.Random(seed)

    compliance_yes: list[dict[str, Any]] = []
    compliance_no: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []

    for ex in examples:
        if ex["question_type"] == "per_component_compliance":
            if ex["answer"] == "Yes":
                compliance_yes.append(ex)
            else:
                compliance_no.append(ex)
        else:
            other.append(ex)

    n_yes = len(compliance_yes)
    n_no = len(compliance_no)

    if n_no == 0 or n_no >= n_yes * target_ratio / (1 - target_ratio):
        return examples

    n_no_target = int(n_yes * target_ratio / (1 - target_ratio))
    n_to_add = n_no_target - n_no

    extra_no = rng.choices(compliance_no, k=n_to_add)

    result = compliance_yes + compliance_no + extra_no + other
    rng.shuffle(result)

    return result


def format_chat_mistral(example: dict[str, Any]) -> dict[str, Any]:
    """Format a flat example as a Mistral chat message for finetuning.

    Produces the OpenAI-style messages format that Mistral's finetuning
    API and HuggingFace trainers expect. The user turn contains the
    image reference and the prompt (instruction + spec + question).
    The assistant turn contains the answer followed by reasoning, which
    teaches the model to explain its compliance judgments.

    Args:
        example: A flat example dict from flatten_records.

    Returns:
        Dict with keys:
            messages: List of system/user/assistant message dicts.
            image_path: Path to the image file (for the training framework
                to load during batching).
    """
    prompt = prompt_builder(
        example["question_type"], example["spec_text"], example["question"]
    )

    answer = example["answer"]
    reasoning = example["reasoning"]

    if isinstance(answer, list):
        if answer:
            assistant_content = "\n".join(answer) + "\n" + reasoning
        else:
            assistant_content = "No violations found. " + reasoning
    else:
        assistant_content = str(answer) + ". " + reasoning

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"file://{example['image_path']}"},
                },
                {"type": "text", "text": prompt},
            ],
        },
        {"role": "assistant", "content": assistant_content},
    ]

    return {
        "messages": messages,
        "image_path": example["image_path"],
    }


def prepare_dataset(
    dataset_path: str,
    dataset_dir: str | None = None,
    val_ratio: float = 0.15,
    question_types: list[str] | None = None,
    oversample: bool = True,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Full pipeline: load, split, flatten, oversample, format.

    Convenience function that chains all steps together.

    Args:
        dataset_path: Path to dataset.jsonl.
        dataset_dir: Root directory of the dataset. If None, inferred
            from dataset_path.
        val_ratio: Fraction of records for validation.
        question_types: Optional filter for question types.
        oversample: Whether to oversample minority compliance class.
        seed: Random seed.

    Returns:
        Tuple of (train_chat, val_chat) where each is a list of
        formatted chat message dicts ready for training.
    """
    if dataset_dir is None:
        dataset_dir = os.path.dirname(dataset_path)

    train_records, val_records = load_and_split(dataset_path, val_ratio, seed)

    train_examples = flatten_records(train_records, dataset_dir, question_types)
    val_examples = flatten_records(val_records, dataset_dir, question_types)

    if oversample:
        train_examples = oversample_minority(train_examples, seed=seed)

    train_chat = [format_chat_mistral(ex) for ex in train_examples]
    val_chat = [format_chat_mistral(ex) for ex in val_examples]

    print(
        f"Train: {len(train_records)} records → "
        f"{len(train_examples)} examples → {len(train_chat)} chat"
    )
    print(
        f"Val:   {len(val_records)} records → "
        f"{len(val_examples)} examples → {len(val_chat)} chat"
    )

    return train_chat, val_chat


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test data loader")
    parser.add_argument(
        "--dataset",
        default="../dataset/dataset.jsonl",
        help="Path to dataset JSONL",
    )
    args = parser.parse_args()

    dataset_dir = os.path.dirname(args.dataset)
    train_records, val_records = load_and_split(args.dataset)
    print(f"Records: {len(train_records)} train, {len(val_records)} val")

    train_examples = flatten_records(train_records, dataset_dir)
    val_examples = flatten_records(val_records, dataset_dir)
    print(f"Examples: {len(train_examples)} train, {len(val_examples)} val")

    yes_before = sum(
        1
        for ex in train_examples
        if ex["question_type"] == "per_component_compliance" and ex["answer"] == "Yes"
    )
    no_before = sum(
        1
        for ex in train_examples
        if ex["question_type"] == "per_component_compliance" and ex["answer"] == "No"
    )
    print(
        f"\nBefore oversampling: {yes_before} Yes, {no_before} No "
        f"({no_before / (yes_before + no_before) * 100:.1f}% No)"
    )

    train_examples = oversample_minority(train_examples)

    yes_after = sum(
        1
        for ex in train_examples
        if ex["question_type"] == "per_component_compliance" and ex["answer"] == "Yes"
    )
    no_after = sum(
        1
        for ex in train_examples
        if ex["question_type"] == "per_component_compliance" and ex["answer"] == "No"
    )
    print(
        f"After oversampling:  {yes_after} Yes, {no_after} No "
        f"({no_after / (yes_after + no_after) * 100:.1f}% No)"
    )

    chat = format_chat_mistral(train_examples[0])
    print("\nSample chat message:")
    print(f"  System: {chat['messages'][0]['content'][:80]}...")
    print(f"  User text: {chat['messages'][1]['content'][1]['text'][:80]}...")
    print(f"  Assistant: {chat['messages'][2]['content'][:80]}...")
    print(f"  Image: {chat['image_path']}")
