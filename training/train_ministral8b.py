"""Training script for VLM engineering compliance finetuning.

Fine-tunes Ministral 3 8B (vision-language model) with LoRA adapters
on the synthetic plate compliance dataset using HuggingFace TRL.

The model is distributed as FP8 weights on HuggingFace. On GPUs with
compute capability >= 8.9 (H100, 4090), FP8 runs natively. On older
GPUs (A100), it auto-dequantizes to bf16. Either way, LoRA adapters
are trained in bf16.

Usage:
    python train_ministral8b.py
    python train_ministral8b.py --dataset ../data/train/dataset.jsonl
    python train_ministral8b.py --epochs 3 --lr 2e-4 --batch-size 2

Requirements:
    pip install torch transformers peft trl bitsandbytes accelerate pillow
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import torch
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    FineGrainedFP8Config,
)
from trl import SFTConfig, SFTTrainer

# Add parent directory to path so we can import evaluate.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_loader import (
    ComplianceDataset,
    flatten,
    load_records,
    make_collate_fn,
    oversample_no,
    train_val_split,
)


def load_model_and_processor(
    model_id: str,
) -> tuple[AutoModelForImageTextToText, AutoProcessor]:
    """Load the vision-language model and processor.

    Uses FineGrainedFP8Config to load the model. On GPUs with compute
    capability >= 8.9, FP8 runs natively (~9GB). On older GPUs, it
    auto-dequantizes to bf16 (~17GB).

    Args:
        model_id: HuggingFace model ID.

    Returns:
        Tuple of (model, processor).
    """
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="auto",
        attn_implementation="eager",
        quantization_config=FineGrainedFP8Config(),
    )

    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "right"

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.add_special_tokens(
            {"pad_token": processor.tokenizer.eos_token}
        )

    print(f"Model loaded: {model_id}")
    print(f"Parameters: {model.num_parameters():,}")

    return model, processor


def get_lora_config(
    r: int = 64,
    alpha: int = 128,
    dropout: float = 0.05,
) -> LoraConfig:
    """Create LoRA configuration for Ministral 3.

    Targets all attention projections and MLP layers in both the
    language model and vision encoder. Since the vision encoder uses
    the same layer names (q_proj, k_proj, etc.), LoRA adapters are
    applied to both automatically.

    TODO: Consider separate LoRA ranks for vision vs language layers,
    or freezing vision encoder for a language-only baseline.

    Args:
        r: LoRA rank. Higher = more capacity, more memory.
        alpha: LoRA scaling factor. Typically 2x rank.
        dropout: Dropout on LoRA layers for regularization.

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
    num_epochs: int = 1,
    batch_size: int = 1,
    grad_accum: int = 8,
    learning_rate: float = 2e-4,
    warmup_steps: int = 50,
    max_seq_len: int = 4096,
    save_strategy: str = "epoch",
    logging_steps: int = 10,
) -> SFTConfig:
    """Create SFT training configuration.

    Uses cosine learning rate schedule with linear warmup, 8-bit AdamW
    optimizer, bfloat16 mixed precision, and gradient checkpointing
    to minimize memory usage.

    Args:
        output_dir: Where to save checkpoints.
        num_epochs: Number of training epochs.
        batch_size: Per-device batch size.
        grad_accum: Gradient accumulation steps
            (effective batch = batch_size x grad_accum).
        learning_rate: Peak learning rate.
        warmup_steps: Linear warmup steps.
        max_seq_len: Maximum sequence length.
        save_strategy: When to save checkpoints.
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
        max_length=max_seq_len,
    )


def train(
    dataset_path: str,
    output_dir: str,
    model_id: str = "mistralai/Ministral-3-8B-Instruct-2512",
    num_epochs: int = 1,
    batch_size: int = 1,
    grad_accum: int = 8,
    learning_rate: float = 2e-4,
    lora_r: int = 64,
    lora_alpha: int = 128,
    max_seq_len: int = 4096,
    val_ratio: float = 0.15,
    oversample_ratio: float = 0.0,
    seed: int = 42,
) -> None:
    """Run the full training pipeline.

    1. Load dataset and split into train/val by example
    2. Flatten records into individual question-answer pairs
    3. Optionally oversample 'No' compliance examples
    4. Load model with FP8 (auto-dequantizes on older GPUs)
    5. Attach LoRA adapters to language + vision layers
    6. Train with SFTTrainer
    7. Save adapter weights and processor

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
        max_seq_len: Maximum sequence length.
        val_ratio: Validation split ratio.
        oversample_ratio: Target fraction of No among compliance
            examples. 0.0 means no oversampling.
        seed: Random seed.
    """
    dataset_dir = os.path.dirname(dataset_path)
    records = load_records(dataset_path)
    train_records, val_records = train_val_split(records, val_ratio, seed)

    train_examples = flatten(train_records, dataset_dir)
    val_examples = flatten(val_records, dataset_dir)
    print(f"Train: {len(train_records)} records -> {len(train_examples)} examples")
    print(f"Val:   {len(val_records)} records -> {len(val_examples)} examples")

    if oversample_ratio > 0:
        train_examples = oversample_no(train_examples, oversample_ratio)
        print(f"Train after oversampling: {len(train_examples)} examples")

    train_dataset = ComplianceDataset(train_examples)
    val_dataset = ComplianceDataset(val_examples)

    model, processor = load_model_and_processor(model_id)

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
        "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Per-device batch size"
    )
    parser.add_argument(
        "--grad-accum", type=int, default=8, help="Gradient accumulation steps"
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument(
        "--max-seq-len", type=int, default=4096, help="Maximum sequence length"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15, help="Validation split ratio"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--oversample",
        type=float,
        default=0.0,
        help="Target No ratio for oversampling (0.0 = off, 0.3 = 30%% No)",
    )

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
        max_seq_len=args.max_seq_len,
        val_ratio=args.val_ratio,
        oversample_ratio=args.oversample,
        seed=args.seed,
    )
