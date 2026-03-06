import argparse
import os

import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType
from trl import SFTConfig, SFTTrainer

from data_loader import (
    load_records,
    train_val_split,
    flatten,
    ComplianceDataset,
    make_collate_fn,
)


def load_model_and_processor(
    model_id: str,
    use_4bit: bool = True,
) -> tuple[AutoModelForImageTextToText, AutoProcessor]:
    """Load the vision-language model and processor.

    Supports 4-bit NF4 quantization via bitsandbytes for fitting
    on a single GPU with reduced memory. Computation remains in
    bfloat16 for numerical stability.

    Args:
        model_id: HuggingFace model ID.
        use_4bit: Whether to load in 4-bit quantization.

    Returns:
        Tuple of (model, processor).
    """
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        quant_config = None

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        quantization_config=quant_config,
    )

    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "right"

    return model, processor

def get_lora_config(
    r: int = 64,
    alpha: int = 128,
    dropout: float = 0.05,
) -> LoraConfig:
    """Create LoRA configuration for Ministral 3.
     
    TODO: ADD VISION ENCODER BEING TARGETED ALSO CHANGE README REGARDING THAT PERHAPS

    Targets all attention projections and MLP layers for maximum
    adaptation capability while keeping trainable parameters small.

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
        grad_accum: Gradient accumulation steps (effective batch = batch_size x grad_accum).
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
        max_seq_length=max_seq_len,
    )


def train(
    dataset_path: str,
    output_dir: str,
    model_id: str = "mistralai/Ministral-3-8B-Instruct-2512",
    num_epochs: int = 1,
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

    Loads data, splits into train/val, loads the quantized model,
    attaches LoRA adapters, trains with SFTTrainer, and saves
    the adapter weights.

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
    records = load_records(dataset_path)
    train_records, val_records = train_val_split(records, val_ratio, seed)

    
    train_examples = flatten(train_records, dataset_dir)
    val_examples = flatten(val_records, dataset_dir)
    print(f"Train: {len(train_records)} records -> {len(train_examples)} examples")
    print(f"Val:   {len(val_records)} records -> {len(val_examples)} examples")

    
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