"""Training script for VLM engineering compliance finetuning.

Fine-tunes Ministral 3 8B (vision-language model) with LoRA adapters
on the synthetic plate compliance dataset using HuggingFace TRL.

Usage::

    python train.py
    python train.py --dataset ../data/train/dataset.jsonl
    python train.py --model mistralai/Ministral-3-8B-Instruct-2512
    python train.py --epochs 3 --lr 2e-4 --batch-size 2

Requirements::

    pip install torch transformers peft trl bitsandbytes
    pip install git+https://github.com/huggingface/transformers.git
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import torch
from peft import LoraConfig, TaskType
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from data_loader import flatten_records, format_chat_mistral, load_and_split


class ComplianceDataset(torch.utils.data.Dataset):
    """PyTorch dataset wrapping formatted chat examples with images.

    Each item returns a dict with 'messages' (chat format) and 'image'
    (PIL Image), which the custom collator processes into model inputs.

    Args:
        examples: List of flat example dicts from flatten_records.
    """

    def __init__(self, examples: list[dict[str, Any]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        chat = format_chat_mistral(ex)

        image = Image.open(ex["image_path"]).convert("RGB")

        return {
            "messages": chat["messages"],
            "image": image,
        }


def make_collate_fn(processor: AutoProcessor, max_seq_len: int = 2048) -> Any:
    """Create a collate function that batches images and chat text together.

    The collator applies the chat template, tokenizes text, processes
    images, and creates labels with padding/image tokens masked out.

    Args:
        processor: The model's processor (tokenizer + image processor).
        max_seq_len: Maximum sequence length for truncation.

    Returns:
        Collate function compatible with PyTorch DataLoader.
    """
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id
    image_token_id = getattr(processor, "image_token_id", None)

    def collate_fn(
        examples: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        images = [ex["image"] for ex in examples]
        conversations = [ex["messages"] for ex in examples]

        chat_texts = processor.apply_chat_template(
            conversations,
            add_generation_prompt=False,
            tokenize=False,
        )

        batch = processor(
            text=chat_texts,
            images=images,
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


def load_model_and_processor(
    model_id: str,
    use_4bit: bool = True,
) -> tuple[AutoModelForImageTextToText, AutoProcessor]:
    """Load the vision-language model and processor.

    Supports 4-bit quantization via bitsandbytes for fitting on
    consumer GPUs (24 GB VRAM).

    Args:
        model_id: HuggingFace model ID.
        use_4bit: Whether to load in 4-bit quantization.

    Returns:
        Tuple of (model, processor).
    """
    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "eager",
    }

    if use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    print(f"Loading model: {model_id}")
    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)

    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "right"

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.add_special_tokens(
            {"pad_token": processor.tokenizer.eos_token}
        )

    print(f"Model loaded. Parameters: {model.num_parameters():,}")
    return model, processor


def get_lora_config(
    r: int = 64,
    alpha: int = 128,
    dropout: float = 0.05,
) -> LoraConfig:
    """Create LoRA configuration for Ministral 3.

    Targets all attention projections and MLP layers for maximum
    adaptation capability while keeping trainable parameters small.

    Args:
        r: LoRA rank (higher = more capacity, more memory).
        alpha: LoRA scaling factor (typically 2× rank).
        dropout: Dropout on LoRA layers.

    Returns:
        LoraConfig for peft.
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def get_training_config(
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 2,
    grad_accum: int = 4,
    learning_rate: float = 2e-4,
    warmup_steps: int = 50,
    max_seq_len: int = 2048,
    save_strategy: str = "epoch",
    logging_steps: int = 10,
) -> SFTConfig:
    """Create SFT training configuration.

    Args:
        output_dir: Where to save checkpoints.
        num_epochs: Number of training epochs.
        batch_size: Per-device batch size.
        grad_accum: Gradient accumulation steps
            (effective batch = batch_size × grad_accum).
        learning_rate: Peak learning rate.
        warmup_steps: Linear warmup steps.
        max_seq_len: Maximum sequence length.
        save_strategy: When to save checkpoints ('epoch' or 'steps').
        logging_steps: Log every N steps.

    Returns:
        SFTConfig for TRL trainer.
    """
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_8bit",
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        bf16=True,
        fp16=False,
        max_grad_norm=0.3,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        eval_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        label_names=["labels"],
        max_seq_length=max_seq_len,
    )


def train(
    dataset_path: str,
    output_dir: str,
    model_id: str = "mistralai/Ministral-3-8B-Instruct-2512",
    num_epochs: int = 3,
    batch_size: int = 2,
    grad_accum: int = 4,
    learning_rate: float = 2e-4,
    lora_r: int = 64,
    lora_alpha: int = 128,
    use_4bit: bool = True,
    max_seq_len: int = 2048,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    """Run the full training pipeline.

    Steps: load and split data, flatten into individual examples, load
    model with quantization, attach LoRA adapters, train with SFTTrainer,
    save adapter weights.

    Args:
        dataset_path: Path to dataset.jsonl.
        output_dir: Where to save the trained adapter.
        model_id: HuggingFace model ID.
        num_epochs: Number of training epochs.
        batch_size: Per-device batch size.
        grad_accum: Gradient accumulation steps.
        learning_rate: Peak learning rate.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha.
        use_4bit: Whether to use 4-bit quantization.
        max_seq_len: Maximum sequence length.
        val_ratio: Validation split ratio.
        seed: Random seed.
    """
    dataset_dir = os.path.dirname(dataset_path)
    train_records, val_records = load_and_split(dataset_path, val_ratio, seed)
    train_examples = flatten_records(train_records, dataset_dir)
    val_examples = flatten_records(val_records, dataset_dir)

    print(f"Train: {len(train_records)} records → " f"{len(train_examples)} examples")
    print(f"Val:   {len(val_records)} records → " f"{len(val_examples)} examples")

    train_yes = sum(
        1
        for ex in train_examples
        if ex["question_type"] == "per_component_compliance" and ex["answer"] == "Yes"
    )
    train_no = sum(
        1
        for ex in train_examples
        if ex["question_type"] == "per_component_compliance" and ex["answer"] == "No"
    )
    if train_yes + train_no > 0:
        print(
            f"Compliance balance: {train_yes} Yes, {train_no} No "
            f"({train_no / (train_yes + train_no) * 100:.1f}% No)"
        )

    train_dataset = ComplianceDataset(train_examples)
    val_dataset = ComplianceDataset(val_examples)

    model, processor = load_model_and_processor(model_id, use_4bit)

    lora_config = get_lora_config(r=lora_r, alpha=lora_alpha)

    training_config = get_training_config(
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        learning_rate=learning_rate,
        max_seq_len=max_seq_len,
    )

    collate_fn = make_collate_fn(processor, max_seq_len)

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        processing_class=processor,
        data_collator=collate_fn,
    )

    print("\nStarting training...")
    trainer.train()

    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"\nAdapter saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune Ministral 3 8B for compliance checking"
    )
    parser.add_argument(
        "--dataset",
        default="../data/train/dataset.jsonl",
        help="Path to training dataset.jsonl",
    )
    parser.add_argument(
        "--output",
        default="../results/finetuned/ministral-8b-lora",
        help="Output directory for adapter weights",
    )
    parser.add_argument(
        "--model",
        default="mistralai/Ministral-3-8B-Instruct-2512",
        help="Base model HuggingFace ID",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Per-device batch size"
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (needs more VRAM)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        output_dir=args.output,
        model_id=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_4bit=not args.no_4bit,
        max_seq_len=args.max_seq_len,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
