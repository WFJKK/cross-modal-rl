"""
Evaluation script for VLM engineering compliance dataset.

Two modes:
    python evaluate.py run   --model mistral-small-latest --dataset dataset/dataset.jsonl
    python evaluate.py score --predictions predictions.jsonl

The 'run' command sends questions to a model API and saves predictions.
The 'score' command reads predictions and computes all metrics.

Metrics:
    1. Compliance accuracy (overall + 3x3 annotation×complexity grid)
    2. Measurement MAE (overall + by annotation + by measurement type)
    3. Audit F1 (precision, recall, F1)
    4. Counterfactual error (absolute error in mm)
    5. Rule selection accuracy (conditional complexity only)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from typing import Any
import random

import requests
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText


# ============================================================
# Constants
# ============================================================

SYSTEM_PROMPT = (
    "You are an engineering compliance checker. "
    "You are given a specification document and a technical drawing of a plate with holes. "
    "Answer the question precisely based on the image and specification."
)

API_CONFIG: dict[str, dict[str, str]] = {
    "mistral": {
        "url": "https://api.mistral.ai/v1/chat/completions",
        "env_key": "MISTRAL_API_KEY",
        "format": "openai",
    },
    "openai": {
        "url": "https://api.openai.com/v1/chat/completions",
        "env_key": "OPENAI_API_KEY",
        "format": "openai",
    },
    "anthropic": {
        "url": "https://api.anthropic.com/v1/messages",
        "env_key": "ANTHROPIC_API_KEY",
        "format": "anthropic",
    },
}

MEASUREMENT_TYPES: list[str] = [
    "measurement_diameter",
    "measurement_edge_distance",
    "measurement_hole_to_hole",
    "measurement_plate_dims",
]

ANNOTATION_LEVELS: list[str] = ["full", "partial", "minimal"]
COMPLEXITY_LEVELS: list[str] = ["simple", "multi_rule", "conditional"]


# ============================================================
# Parsers
# ============================================================

def extract_yes_no(sentence: str) -> str | None:
    """Extract a Yes/No answer from free-text model response.

    Uses a three-stage strategy:
    1. Look for a 'Final Answer: Yes/No' pattern (self-correcting models).
    2. Check if the first line is exactly 'yes' or 'no' (expected format).
    3. Fall back to checking if the response starts with 'yes' or 'no'.

    Args:
        sentence: Raw model response text.

    Returns:
        'Yes', 'No', or None if unparseable.
    """
    lower = sentence.strip().lower()
    match = re.search(r"final answer[\s*:]*\s*(yes|no)", lower)
    if match:
        return "Yes" if match.group(1) == "yes" else "No"
    first_line = lower.split("\n")[0].strip().rstrip(".")
    if first_line == "yes":
        return "Yes"
    if first_line == "no":
        return "No"
    if lower.startswith("yes"):
        return "Yes"
    if lower.startswith("no"):
        return "No"
    return None


def extract_number(sentence: str) -> float | None:
    """Extract a numeric measurement from free-text model response.

    Looks for a number followed by 'mm'. Returns None if no match found.
    Designed for measurement questions where the prompt instructs the model
    to answer as a number followed by mm.

    Args:
        sentence: Raw model response text.

    Returns:
        The extracted float value, or None if unparseable.
    """
    mm_match = re.search(r'(\d+\.?\d*)\s*mm', sentence)
    if mm_match:
        return float(mm_match.group(1))
    return None


def extract_dimensions(sentence: str) -> tuple[float, float] | None:
    """Extract two dimensions (width × height) from a plate dimensions response.

    Handles formats like '152 mm × 91 mm', '152.0×91.0mm', '152mm x 91mm'.
    Returns dimensions as a sorted tuple so order doesn't matter.

    Args:
        sentence: Raw model response or ground truth string.

    Returns:
        Sorted tuple of (smaller, larger) dimensions, or None if < 2 numbers found.
    """
    # First try: numbers near mm or × separators
    numbers = re.findall(r'(\d+\.?\d*)\s*(?:mm|×|x|X|\*)', sentence)
    if len(numbers) < 2:
        # Fallback: any two numbers followed by mm
        numbers = re.findall(r'(\d+\.?\d*)\s*mm', sentence)
    if len(numbers) < 2:
        # Last resort: find numbers around × or x separator
        numbers = re.findall(r'(\d+\.?\d*)\s*[×xX\*]\s*(\d+\.?\d*)', sentence)
        if numbers:
            numbers = list(numbers[0])
    if len(numbers) >= 2:
        return tuple(sorted([float(numbers[0]), float(numbers[1])]))
    return None


def extract_violations(sentence: str) -> set[tuple[str, str]]:
    """Extract (hole_id, rule_id) pairs from a violation list response.

    Matches patterns where H# and R# appear within 20 characters of each
    other, handling formats like 'H1: Rule R2 violation', 'H1 violates R2',
    'H1 — R2', etc.

    Args:
        sentence: Raw model response text.

    Returns:
        Set of (hole_id, rule_id) tuples, e.g. {('H1', 'R2'), ('H3', 'R4')}.
    """
    pattern = r'(H\d+)\D{0,20}(R\d+)'
    matches = re.findall(pattern, sentence, re.IGNORECASE)
    return {(h.upper(), r.upper()) for h, r in matches}


# ============================================================
# Prompt builder
# ============================================================

QUESTION_PREFIXES: dict[str, str] = {
    "per_component_compliance": "Your first line must be exactly Yes or No with no other text. Then explain your reasoning on the following lines.",
    "full_audit": "List all violations in the format: H#: Rule R# violation. If none, say 'No violations found.'",
    "measurement_diameter": "Give your answer as a number followed by mm.",
    "measurement_edge_distance": "Give your answer as a number followed by mm.",
    "measurement_hole_to_hole": "Give your answer as a number followed by mm.",
    "measurement_plate_dims": "Give your answer as a number followed by mm.",
    "rule_selection": "State the applicable parameter value.",
    "counterfactual": "Give your answer as a number followed by mm.",
}


def prompt_builder(question_type: str, spec_text: str, question: str) -> str:
    """Build the user prompt from question type, specification, and question.

    Combines an instruction prefix (based on question type) with the
    specification document and the actual question. Does not include the
    system prompt — that is handled separately by the model interface.

    Args:
        question_type: One of the keys in QUESTION_PREFIXES.
        spec_text: The specification document text from the dataset.
        question: The question string from the dataset.

    Returns:
        Formatted prompt string ready to send to the model.

    Raises:
        ValueError: If question_type is not recognized.
    """
    if question_type not in QUESTION_PREFIXES:
        raise ValueError(f"Unknown question type: {question_type}")

    prefix = QUESTION_PREFIXES[question_type]
    return f"{prefix}\n\nSpecification:\n{spec_text}\n\nQuestion:\n{question}"


# ============================================================
# Model interface
# ============================================================

def call_api(image_path: str, prompt: str, model: str, provider: str = "mistral") -> str:
    """Send a question with an image to a model API and return the response.

    Supports Mistral/OpenAI format and Anthropic format. Reads the API key
    from the environment variable specified in API_CONFIG. Retries up to 3
    times on rate limit errors (HTTP 429).

    Args:
        image_path: Path to the PNG image file.
        prompt: The formatted prompt from prompt_builder.
        model: Model identifier string (e.g. 'mistral-small-latest').
        provider: One of 'mistral', 'openai', 'anthropic'.

    Returns:
        The model's text response.

    Raises:
        ValueError: If provider is unknown or API key is not set.
        RuntimeError: If the API returns an error or rate limits after 3 retries.
    """
    if provider not in API_CONFIG:
        raise ValueError(f"Unknown provider: {provider}")

    config = API_CONFIG[provider]
    api_key = os.environ.get(config["env_key"])
    if not api_key:
        raise ValueError(f"Set {config['env_key']} environment variable")

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    if config["format"] == "openai":
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": 1024,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        }

    elif config["format"] == "anthropic":
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        body = {
            "model": model,
            "max_tokens": 1024,
            "system": SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        }

    for attempt in range(3):
        response = requests.post(config["url"], headers=headers, json=body)

        if response.status_code == 200:
            data = response.json()
            if config["format"] == "openai":
                return data["choices"][0]["message"]["content"]
            elif config["format"] == "anthropic":
                return data["content"][0]["text"]
        elif response.status_code == 429:
            time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"API error {response.status_code}: {response.text}")

    raise RuntimeError("Rate limited after 3 retries")


# ============================================================
# Local model cache (loaded once, reused across questions)
# ============================================================

_local_model: Any = None
_local_processor: Any = None


def _load_local_model(adapter_path: str) -> None:
    """Load base model with LoRA adapter. Called once, result cached.

    Reads the adapter config to find the base model ID, loads it with
    FineGrainedFP8Config (auto-dequantizes to bf16 on older GPUs),
    then attaches the LoRA adapter on top.

    Args:
        adapter_path: Path to the LoRA adapter directory.
    """
    global _local_model, _local_processor

    if _local_model is not None:
        return

    from transformers import AutoProcessor, FineGrainedFP8Config
    from peft import PeftModel

    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(adapter_config_path) as f:
        adapter_config = json.load(f)
    base_model_id = adapter_config["base_model_name_or_path"]

    print(f"Loading base model: {base_model_id}")
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        device_map="auto",
        attn_implementation="eager",
        quantization_config=FineGrainedFP8Config(),
    )

    print(f"Loading LoRA adapter: {adapter_path}")
    _local_model = PeftModel.from_pretrained(base_model, adapter_path)
    _local_model.eval()

    _local_processor = AutoProcessor.from_pretrained(adapter_path)
    if _local_processor.tokenizer.pad_token is None:
        _local_processor.tokenizer.add_special_tokens(
            {"pad_token": _local_processor.tokenizer.eos_token}
        )

    print("Model ready for inference")


def call_local(image_path: str, prompt: str, model: str) -> str:
    """Run inference with a local finetuned model.

    Loads the base model + LoRA adapter on first call (cached for
    subsequent calls). Processes the image and prompt, generates a
    response using greedy decoding.

    Args:
        image_path: Path to the PNG image file.
        prompt: The formatted prompt from prompt_builder.
        model: Path to the LoRA adapter directory.

    Returns:
        The model's text response.
    """
    _load_local_model(model)

    image = Image.open(image_path).convert("RGB")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    text = _local_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = _local_processor(
        text=text,
        images=[image],
        return_tensors="pt",
    ).to(_local_model.device)

    with torch.no_grad():
        output_ids = _local_model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    response = _local_processor.tokenizer.decode(
        new_tokens, skip_special_tokens=True
    )

    return response


def call_model(
    image_path: str, prompt: str, backend: str, model: str, provider: str = "mistral"
) -> str:
    """Route a model call to the appropriate backend.

    Args:
        image_path: Path to the PNG image file.
        prompt: The formatted prompt from prompt_builder.
        backend: 'api' for remote API or 'local' for local inference.
        model: Model identifier or local path.
        provider: API provider (only used when backend='api').

    Returns:
        The model's text response.

    Raises:
        ValueError: If backend is unknown.
    """
    if backend == "api":
        return call_api(image_path, prompt, model, provider)
    elif backend == "local":
        return call_local(image_path, prompt, model)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ============================================================
# Prediction runner
# ============================================================

def load_completed_examples(output_path: str) -> set[str]:
    """Load example_ids that have already been predicted, for resume capability.

    Args:
        output_path: Path to the existing predictions JSONL file.

    Returns:
        Set of example_ids that are already in the output file.
    """
    completed: set[str] = set()
    if not os.path.exists(output_path):
        return completed
    with open(output_path) as f:
        for line in f:
            try:
                pred = json.loads(line)
                completed.add(pred["example_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def run_predictions(
    dataset_path: str,
    output_path: str,
    backend: str,
    model: str,
    provider: str = "mistral",
    max_examples: int | None = None,
    question_types: list[str] | None = None,
) -> None:
    """Run model predictions on the dataset and save to a JSONL file.

    Loads examples from the dataset, sends each question to the model,
    and writes predictions with ground truth and metadata. Supports
    resuming from a partial run — skips examples that are already in the
    output file. Shuffles with a fixed seed for reproducibility.

    Args:
        dataset_path: Path to dataset/dataset.jsonl.
        output_path: Path to write predictions.jsonl.
        backend: 'api' or 'local'.
        model: Model identifier or local path.
        provider: API provider name (for backend='api').
        max_examples: Maximum number of examples to process (None for all).
        question_types: List of question types to include (None for all).
    """
    with open(dataset_path, "r") as f:
        data = [json.loads(line) for line in f]

    random.seed(42)
    random.shuffle(data)

    if max_examples is not None:
        data = data[:max_examples]

    dataset_dir = os.path.dirname(dataset_path)

    # Resume: skip already-completed examples
    completed = load_completed_examples(output_path)
    if completed:
        print(f"Resuming: {len(completed)} examples already done, skipping them.")

    # Append if resuming, write fresh otherwise
    mode = "a" if completed else "w"
    start_time = time.time()
    total_questions = 0

    with open(output_path, mode) as out:
        for i, record in enumerate(data):
            if record["example_id"] in completed:
                continue

            image_path = os.path.join(dataset_dir, record["image"])
            spec_text = record["spec_text"]
            metadata = record["metadata"]

            for j, question in enumerate(record["questions"]):
                if question_types and question["type"] not in question_types:
                    continue

                prompt = prompt_builder(question["type"], spec_text, question["question"])
                response = call_model(image_path, prompt, backend, model, provider)

                prediction = {
                    "example_id": record["example_id"],
                    "question_index": j,
                    "question_type": question["type"],
                    "response": response,
                    "ground_truth": question["answer"],
                    "rule_type": question.get("rule_type", None),
                    "annotation_level": metadata["annotation_level"],
                    "rule_complexity": metadata["rule_complexity"],
                }
                out.write(json.dumps(prediction) + "\n")
                out.flush()
                total_questions += 1

            elapsed = time.time() - start_time
            done = i + 1 - len(completed)
            remaining = len(data) - len(completed) - done
            if done > 0:
                avg_time = elapsed / done
                eta = avg_time * remaining
                eta_min = eta / 60
            else:
                eta_min = 0.0

            if (i + 1) % 10 == 0:
                print(
                    f"[{i + 1}/{len(data)}] {total_questions} questions | "
                    f"{elapsed:.0f}s elapsed | ~{eta_min:.1f}min remaining"
                )

    elapsed = time.time() - start_time
    print(f"\nDone. {total_questions} predictions saved to {output_path} ({elapsed:.0f}s)")


# ============================================================
# Scorers
# ============================================================

def score_compliance(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Score per-component compliance predictions (Yes/No accuracy).

    Computes overall accuracy and breakdowns by annotation level,
    rule complexity, rule type, and ground truth label (bias detection).

    Args:
        predictions: List of prediction dicts from predictions.jsonl.

    Returns:
        Dict with keys: overall, by_annotation, by_complexity, by_rule_type,
        by_ground_truth, errors.
    """
    correct = 0
    total = 0
    unparseable = 0
    by_annotation: dict[str, dict[str, int]] = {}
    by_complexity: dict[str, dict[str, int]] = {}
    by_rule_type: dict[str, dict[str, int]] = {}
    by_ground_truth: dict[str, dict[str, int]] = {
        "Yes": {"correct": 0, "total": 0},
        "No": {"correct": 0, "total": 0},
    }
    errors: list[dict[str, Any]] = []

    for p in predictions:
        if p["question_type"] != "per_component_compliance":
            continue

        parsed = extract_yes_no(p["response"])
        if parsed is None:
            unparseable += 1
            continue

        total += 1
        is_correct = int(parsed == p["ground_truth"])
        correct += is_correct

        if not is_correct:
            errors.append({
                "example_id": p["example_id"],
                "question_index": p["question_index"],
                "predicted": parsed,
                "ground_truth": p["ground_truth"],
                "response": p["response"][:200],
                "annotation_level": p["annotation_level"],
                "rule_complexity": p["rule_complexity"],
            })

        # By annotation level
        annot = p["annotation_level"]
        if annot not in by_annotation:
            by_annotation[annot] = {"correct": 0, "total": 0}
        by_annotation[annot]["correct"] += is_correct
        by_annotation[annot]["total"] += 1

        # By complexity
        comp = p["rule_complexity"]
        if comp not in by_complexity:
            by_complexity[comp] = {"correct": 0, "total": 0}
        by_complexity[comp]["correct"] += is_correct
        by_complexity[comp]["total"] += 1

        # By rule type
        rtype = p.get("rule_type")
        if rtype:
            if rtype not in by_rule_type:
                by_rule_type[rtype] = {"correct": 0, "total": 0}
            by_rule_type[rtype]["correct"] += is_correct
            by_rule_type[rtype]["total"] += 1

        # By ground truth label
        gt = p["ground_truth"]
        if gt in by_ground_truth:
            by_ground_truth[gt]["correct"] += is_correct
            by_ground_truth[gt]["total"] += 1

    return {
        "overall": {"correct": correct, "total": total, "unparseable": unparseable},
        "by_annotation": by_annotation,
        "by_complexity": by_complexity,
        "by_rule_type": by_rule_type,
        "by_ground_truth": by_ground_truth,
        "errors": errors,
    }


def score_measurements(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Score measurement extraction predictions (MAE in mm).

    Computes mean absolute error overall and broken down by annotation
    level and measurement type.

    Args:
        predictions: List of prediction dicts from predictions.jsonl.

    Returns:
        Dict with keys: overall, by_annotation, by_type, errors.
    """
    all_errors: list[float] = []
    by_annotation: dict[str, list[float]] = {}
    by_type: dict[str, list[float]] = {}
    unparseable = 0
    errors: list[dict[str, Any]] = []

    for p in predictions:
        if p["question_type"] not in MEASUREMENT_TYPES:
            continue

        # Special handling for plate dimensions (two values)
        if p["question_type"] == "measurement_plate_dims":
            pred_dims = extract_dimensions(p["response"])
            true_dims = extract_dimensions(str(p["ground_truth"]))
            if pred_dims is None:
                unparseable += 1
                continue
            if true_dims is None:
                continue
            # Average error across both dimensions
            error = (abs(pred_dims[0] - true_dims[0]) + abs(pred_dims[1] - true_dims[1])) / 2
        else:
            predicted = extract_number(p["response"])
            if predicted is None:
                unparseable += 1
                continue

            true_value = extract_number(str(p["ground_truth"]))
            if true_value is None:
                continue

            error = abs(predicted - true_value)
        all_errors.append(error)

        if error > 2.0:
            if p["question_type"] == "measurement_plate_dims":
                errors.append({
                    "example_id": p["example_id"],
                    "question_type": p["question_type"],
                    "predicted": list(pred_dims),
                    "true_value": list(true_dims),
                    "error": round(error, 2),
                    "response": p["response"][:200],
                    "annotation_level": p["annotation_level"],
                })
            else:
                errors.append({
                    "example_id": p["example_id"],
                    "question_type": p["question_type"],
                    "predicted": predicted,
                    "true_value": true_value,
                    "error": round(error, 2),
                    "response": p["response"][:200],
                    "annotation_level": p["annotation_level"],
                })

        annot = p["annotation_level"]
        if annot not in by_annotation:
            by_annotation[annot] = []
        by_annotation[annot].append(error)

        mtype = p["question_type"]
        if mtype not in by_type:
            by_type[mtype] = []
        by_type[mtype].append(error)

    def mae(err_list: list[float]) -> float | None:
        if not err_list:
            return None
        return round(sum(err_list) / len(err_list), 2)

    return {
        "overall": {"mae": mae(all_errors), "count": len(all_errors), "unparseable": unparseable},
        "by_annotation": {k: {"mae": mae(v), "count": len(v)} for k, v in by_annotation.items()},
        "by_type": {k: {"mae": mae(v), "count": len(v)} for k, v in by_type.items()},
        "errors": errors,
    }


def score_audit(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Score full audit predictions (precision, recall, F1).

    For each example, compares the set of predicted violations to the
    ground truth set. Precision catches hallucinated violations, recall
    catches missed ones.

    Args:
        predictions: List of prediction dicts from predictions.jsonl.

    Returns:
        Dict with keys: precision, recall, f1, count, unparseable, errors.
    """
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    unparseable = 0
    errors: list[dict[str, Any]] = []

    for p in predictions:
        if p["question_type"] != "full_audit":
            continue

        predicted_violations = extract_violations(p["response"])

        # If model produced no violations and didn't say "no violations", mark unparseable
        if not predicted_violations and "no violation" not in p["response"].lower():
            unparseable += 1
            continue

        # Ground truth is a list of strings like "H1: Rule R2 violation"
        gt_violations: set[tuple[str, str]] = set()
        gt_list = p["ground_truth"]
        if isinstance(gt_list, list):
            for item in gt_list:
                parsed = extract_violations(item)
                gt_violations.update(parsed)

        # Both empty — correct
        if len(gt_violations) == 0 and len(predicted_violations) == 0:
            precisions.append(1.0)
            recalls.append(1.0)
            f1s.append(1.0)
            continue

        # Model found nothing but there were violations
        if len(predicted_violations) == 0:
            precisions.append(0.0)
            recalls.append(0.0)
            f1s.append(0.0)
            errors.append({
                "example_id": p["example_id"],
                "predicted": [],
                "ground_truth": sorted(gt_violations),
                "issue": "missed_all",
            })
            continue

        # Model found violations but there were none
        if len(gt_violations) == 0:
            precisions.append(0.0)
            recalls.append(1.0)
            f1s.append(0.0)
            errors.append({
                "example_id": p["example_id"],
                "predicted": sorted(predicted_violations),
                "ground_truth": [],
                "issue": "all_hallucinated",
            })
            continue

        true_positives = len(predicted_violations & gt_violations)
        precision = true_positives / len(predicted_violations)
        recall = true_positives / len(gt_violations)

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        if f1 < 1.0:
            errors.append({
                "example_id": p["example_id"],
                "predicted": sorted(predicted_violations),
                "ground_truth": sorted(gt_violations),
                "precision": round(precision, 2),
                "recall": round(recall, 2),
            })

    def avg(lst: list[float]) -> float | None:
        if not lst:
            return None
        return round(sum(lst) / len(lst), 3)

    return {
        "precision": avg(precisions),
        "recall": avg(recalls),
        "f1": avg(f1s),
        "count": len(f1s),
        "unparseable": unparseable,
        "errors": errors,
    }


def score_counterfactual(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Score counterfactual predictions (absolute error in mm).

    Counterfactual questions ask 'What minimum value would H# need to comply?'
    The model must compute the correct threshold via backward reasoning.

    Args:
        predictions: List of prediction dicts from predictions.jsonl.

    Returns:
        Dict with keys: overall, by_annotation, errors.
    """
    all_errors: list[float] = []
    by_annotation: dict[str, list[float]] = {}
    unparseable = 0
    errors: list[dict[str, Any]] = []

    for p in predictions:
        if p["question_type"] != "counterfactual":
            continue

        predicted = extract_number(p["response"])
        if predicted is None:
            unparseable += 1
            continue

        true_value = extract_number(str(p["ground_truth"]))
        if true_value is None:
            continue

        error = abs(predicted - true_value)
        all_errors.append(error)

        if error > 1.0:
            errors.append({
                "example_id": p["example_id"],
                "predicted": predicted,
                "true_value": true_value,
                "error": round(error, 2),
                "response": p["response"][:200],
                "annotation_level": p["annotation_level"],
            })

        annot = p["annotation_level"]
        if annot not in by_annotation:
            by_annotation[annot] = []
        by_annotation[annot].append(error)

    def mae(err_list: list[float]) -> float | None:
        if not err_list:
            return None
        return round(sum(err_list) / len(err_list), 2)

    return {
        "overall": {"mae": mae(all_errors), "count": len(all_errors), "unparseable": unparseable},
        "by_annotation": {k: {"mae": mae(v), "count": len(v)} for k, v in by_annotation.items()},
        "errors": errors,
    }


def score_rule_selection(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Score rule selection predictions (accuracy).

    Tests whether the model can correctly parse the spec to find applicable
    parameters. Only applies to conditional complexity examples.

    Args:
        predictions: List of prediction dicts from predictions.jsonl.

    Returns:
        Dict with keys: correct, total, unparseable, accuracy.
    """
    correct = 0
    total = 0
    unparseable = 0

    for p in predictions:
        if p["question_type"] != "rule_selection":
            continue

        # Try to extract a number from both predicted and ground truth
        predicted = extract_number(p["response"])
        true_value = extract_number(str(p["ground_truth"]))

        if predicted is None:
            unparseable += 1
            continue

        if true_value is None:
            continue

        total += 1
        # Allow small floating point tolerance
        if abs(predicted - true_value) < 0.01:
            correct += 1

    accuracy = correct / total if total > 0 else None
    return {
        "correct": correct,
        "total": total,
        "unparseable": unparseable,
        "accuracy": round(accuracy, 3) if accuracy is not None else None,
    }


# ============================================================
# 3×3 Grid: annotation × complexity
# ============================================================

def build_compliance_grid(predictions: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, int]]]:
    """Build the 3×3 compliance accuracy grid (annotation × complexity).

    This is the key diagnostic table: it shows exactly where the model
    fails across the two controllable difficulty axes.

    Args:
        predictions: List of prediction dicts from predictions.jsonl.

    Returns:
        Nested dict: grid[annotation][complexity] = {'correct': int, 'total': int}.
    """
    grid: dict[str, dict[str, dict[str, int]]] = {}
    for annot in ANNOTATION_LEVELS:
        grid[annot] = {}
        for comp in COMPLEXITY_LEVELS:
            grid[annot][comp] = {"correct": 0, "total": 0}

    for p in predictions:
        if p["question_type"] != "per_component_compliance":
            continue

        parsed = extract_yes_no(p["response"])
        if parsed is None:
            continue

        annot = p["annotation_level"]
        comp = p["rule_complexity"]

        if annot in grid and comp in grid[annot]:
            grid[annot][comp]["total"] += 1
            if parsed == p["ground_truth"]:
                grid[annot][comp]["correct"] += 1

    return grid


# ============================================================
# Score all + report
# ============================================================

def score_all(predictions_path: str) -> dict[str, Any]:
    """Load predictions and compute all metrics.

    Args:
        predictions_path: Path to predictions.jsonl.

    Returns:
        Dict with all scored results, ready for printing and saving.
    """
    with open(predictions_path) as f:
        predictions = [json.loads(line) for line in f]

    results: dict[str, Any] = {
        "num_predictions": len(predictions),
        "compliance": score_compliance(predictions),
        "measurements": score_measurements(predictions),
        "audit": score_audit(predictions),
        "counterfactual": score_counterfactual(predictions),
        "rule_selection": score_rule_selection(predictions),
        "grid": build_compliance_grid(predictions),
    }
    return results


def _pct(correct: int, total: int) -> str:
    """Format a fraction as percentage string."""
    if total == 0:
        return "N/A"
    return f"{correct}/{total} = {correct / total:.1%}"


def print_report(results: dict[str, Any]) -> None:
    """Print a formatted evaluation report to stdout.

    Args:
        results: Dict from score_all.
    """
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Total predictions: {results['num_predictions']}")

    # --- Compliance ---
    c = results["compliance"]
    overall = c["overall"]
    print(f"\n--- Compliance Accuracy ---")
    print(f"Overall: {_pct(overall['correct'], overall['total'])}")
    print(f"Unparseable: {overall['unparseable']}")

    print(f"\nBy annotation level:")
    for annot in ANNOTATION_LEVELS:
        if annot in c["by_annotation"]:
            b = c["by_annotation"][annot]
            print(f"  {annot:10s}: {_pct(b['correct'], b['total'])}")

    print(f"\nBy rule complexity:")
    for comp in COMPLEXITY_LEVELS:
        if comp in c["by_complexity"]:
            b = c["by_complexity"][comp]
            print(f"  {comp:12s}: {_pct(b['correct'], b['total'])}")

    if c["by_rule_type"]:
        print(f"\nBy rule type:")
        for rtype, b in sorted(c["by_rule_type"].items()):
            print(f"  {rtype:18s}: {_pct(b['correct'], b['total'])}")

    print(f"\nBias check (accuracy by ground truth label):")
    for label in ["Yes", "No"]:
        b = c["by_ground_truth"][label]
        print(f"  {label:4s}: {_pct(b['correct'], b['total'])}")

    # --- 3×3 Grid ---
    grid = results["grid"]
    print(f"\n--- 3×3 Grid: Annotation × Complexity ---")
    header = f"{'':12s}" + "".join(f"{comp:>14s}" for comp in COMPLEXITY_LEVELS)
    print(header)
    for annot in ANNOTATION_LEVELS:
        row = f"{annot:12s}"
        for comp in COMPLEXITY_LEVELS:
            cell = grid[annot][comp]
            row += f"{'':>2s}{_pct(cell['correct'], cell['total']):>12s}"
        print(row)

    # --- Measurements ---
    m = results["measurements"]
    print(f"\n--- Measurement MAE (mm) ---")
    print(f"Overall: {m['overall']['mae']} mm ({m['overall']['count']} predictions)")
    print(f"Unparseable: {m['overall']['unparseable']}")

    print(f"\nBy annotation level:")
    for annot in ANNOTATION_LEVELS:
        if annot in m["by_annotation"]:
            b = m["by_annotation"][annot]
            print(f"  {annot:10s}: {b['mae']} mm (n={b['count']})")

    print(f"\nBy measurement type:")
    for mtype, b in sorted(m["by_type"].items()):
        short_name = mtype.replace("measurement_", "")
        print(f"  {short_name:18s}: {b['mae']} mm (n={b['count']})")

    # --- Audit ---
    a = results["audit"]
    print(f"\n--- Audit F1 ---")
    print(f"Precision: {a['precision']}")
    print(f"Recall:    {a['recall']}")
    print(f"F1:        {a['f1']}")
    print(f"Count:     {a['count']}")
    print(f"Unparseable: {a['unparseable']}")

    # --- Counterfactual ---
    cf = results["counterfactual"]
    print(f"\n--- Counterfactual MAE (mm) ---")
    print(f"Overall: {cf['overall']['mae']} mm ({cf['overall']['count']} predictions)")
    print(f"Unparseable: {cf['overall']['unparseable']}")

    if cf["by_annotation"]:
        print(f"\nBy annotation level:")
        for annot in ANNOTATION_LEVELS:
            if annot in cf["by_annotation"]:
                b = cf["by_annotation"][annot]
                print(f"  {annot:10s}: {b['mae']} mm (n={b['count']})")

    # --- Rule Selection ---
    rs = results["rule_selection"]
    print(f"\n--- Rule Selection Accuracy ---")
    print(f"Accuracy: {rs['accuracy']}")
    print(f"Count:    {rs['total']}")
    print(f"Unparseable: {rs['unparseable']}")

    print("\n" + "=" * 60)


def save_results(results: dict[str, Any], output_path: str) -> None:
    """Save results dict to JSON, excluding error lists.

    Args:
        results: Dict from score_all.
        output_path: Path to write results.json.
    """
    # Strip error lists for the summary file (they can be large)
    clean = {}
    for key, value in results.items():
        if isinstance(value, dict) and "errors" in value:
            clean[key] = {k: v for k, v in value.items() if k != "errors"}
        else:
            clean[key] = value

    with open(output_path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"Results saved to {output_path}")


def save_errors(results: dict[str, Any], output_path: str) -> None:
    """Save all error cases to a JSONL file for debugging.

    Collects errors from all scorers into one file. Each line has the
    scorer name and the error details.

    Args:
        results: Dict from score_all.
        output_path: Path to write errors.jsonl.
    """
    with open(output_path, "w") as f:
        for scorer_name in ["compliance", "measurements", "audit", "counterfactual"]:
            scorer_results = results.get(scorer_name, {})
            error_list = scorer_results.get("errors", [])
            for error in error_list:
                error["scorer"] = scorer_name
                f.write(json.dumps(error) + "\n")

    total = sum(
        len(results.get(s, {}).get("errors", []))
        for s in ["compliance", "measurements", "audit", "counterfactual"]
    )
    print(f"Saved {total} error cases to {output_path}")


# ============================================================
# Comparison
# ============================================================

def compare_results(path_a: str, path_b: str, label_a: str = "A", label_b: str = "B") -> None:
    """Print a side-by-side comparison of two results.json files.

    Useful for comparing baseline vs finetuned model performance.

    Args:
        path_a: Path to first results.json (e.g. baseline).
        path_b: Path to second results.json (e.g. finetuned).
        label_a: Display label for first results.
        label_b: Display label for second results.
    """
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)

    print("\n" + "=" * 60)
    print(f"COMPARISON: {label_a} vs {label_b}")
    print("=" * 60)

    # Compliance overall
    ca = a["compliance"]["overall"]
    cb = b["compliance"]["overall"]
    acc_a = ca["correct"] / ca["total"] if ca["total"] > 0 else 0
    acc_b = cb["correct"] / cb["total"] if cb["total"] > 0 else 0
    bal_a = ca.get("balanced_accuracy", acc_a)
    bal_b = cb.get("balanced_accuracy", acc_b)
    diff = acc_b - acc_a
    bal_diff = bal_b - bal_a
    print(f"\n--- Compliance Accuracy ---")
    print(f"  {label_a:15s}: {acc_a:.1%}  (balanced: {bal_a:.1%})")
    print(f"  {label_b:15s}: {acc_b:.1%}  (balanced: {bal_b:.1%})")
    print(f"  {'Change':15s}: {diff:+.1%}  (balanced: {bal_diff:+.1%})")

    # By annotation
    print(f"\nBy annotation level:")
    for annot in ANNOTATION_LEVELS:
        ba = a["compliance"].get("by_annotation", {}).get(annot, {"correct": 0, "total": 0})
        bb = b["compliance"].get("by_annotation", {}).get(annot, {"correct": 0, "total": 0})
        acc_a_ann = ba["correct"] / ba["total"] if ba["total"] > 0 else 0
        acc_b_ann = bb["correct"] / bb["total"] if bb["total"] > 0 else 0
        diff = acc_b_ann - acc_a_ann
        print(f"  {annot:10s}: {acc_a_ann:.1%} → {acc_b_ann:.1%} ({diff:+.1%})")

    # By complexity
    print(f"\nBy rule complexity:")
    for comp in COMPLEXITY_LEVELS:
        ba = a["compliance"].get("by_complexity", {}).get(comp, {"correct": 0, "total": 0})
        bb = b["compliance"].get("by_complexity", {}).get(comp, {"correct": 0, "total": 0})
        acc_a_comp = ba["correct"] / ba["total"] if ba["total"] > 0 else 0
        acc_b_comp = bb["correct"] / bb["total"] if bb["total"] > 0 else 0
        diff = acc_b_comp - acc_a_comp
        print(f"  {comp:12s}: {acc_a_comp:.1%} → {acc_b_comp:.1%} ({diff:+.1%})")

    # 3×3 Grid comparison
    if "grid" in a and "grid" in b:
        print(f"\n--- 3×3 Grid Change ---")
        header = f"{'':12s}" + "".join(f"{comp:>14s}" for comp in COMPLEXITY_LEVELS)
        print(header)
        for annot in ANNOTATION_LEVELS:
            row = f"{annot:12s}"
            for comp in COMPLEXITY_LEVELS:
                cell_a = a["grid"].get(annot, {}).get(comp, {"correct": 0, "total": 0})
                cell_b = b["grid"].get(annot, {}).get(comp, {"correct": 0, "total": 0})
                acc_a_cell = cell_a["correct"] / cell_a["total"] if cell_a["total"] > 0 else 0
                acc_b_cell = cell_b["correct"] / cell_b["total"] if cell_b["total"] > 0 else 0
                diff = acc_b_cell - acc_a_cell
                row += f"{'':>2s}{diff:>+11.1%}   "
            print(row)

    # Measurement MAE
    mae_a = a.get("measurements", {}).get("overall", {}).get("mae")
    mae_b = b.get("measurements", {}).get("overall", {}).get("mae")
    if mae_a is not None and mae_b is not None:
        print(f"\n--- Measurement MAE ---")
        print(f"  {label_a:15s}: {mae_a} mm")
        print(f"  {label_b:15s}: {mae_b} mm")
        print(f"  {'Change':15s}: {mae_b - mae_a:+.2f} mm")

    # Audit F1
    f1_a = a.get("audit", {}).get("f1")
    f1_b = b.get("audit", {}).get("f1")
    if f1_a is not None and f1_b is not None:
        print(f"\n--- Audit F1 ---")
        print(f"  {label_a:15s}: {f1_a}")
        print(f"  {label_b:15s}: {f1_b}")
        print(f"  {'Change':15s}: {f1_b - f1_a:+.3f}")

    print("\n" + "=" * 60)


# ============================================================
# CLI
# ============================================================

def main() -> None:
    """Main entry point with subcommands: run, score, compare."""
    parser = argparse.ArgumentParser(
        description="Evaluate VLMs on engineering compliance dataset."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Run model predictions on dataset")
    run_parser.add_argument("--dataset", default="dataset/dataset.jsonl",
                            help="Path to dataset JSONL")
    run_parser.add_argument("--output", default="predictions.jsonl",
                            help="Path to save predictions")
    run_parser.add_argument("--backend", default="api", choices=["api", "local"],
                            help="Model backend")
    run_parser.add_argument("--model", required=True,
                            help="Model identifier (e.g. mistral-small-latest)")
    run_parser.add_argument("--provider", default="mistral",
                            choices=list(API_CONFIG.keys()),
                            help="API provider")
    run_parser.add_argument("--max-examples", type=int, default=None,
                            help="Max examples to process")
    run_parser.add_argument("--types", nargs="+", default=None,
                            help="Question types to evaluate")

    # --- score ---
    score_parser = subparsers.add_parser("score", help="Score existing predictions")
    score_parser.add_argument("--predictions", default="predictions.jsonl",
                              help="Path to predictions JSONL")
    score_parser.add_argument("--results", default="results.json",
                              help="Path to save results JSON")
    score_parser.add_argument("--errors", default="errors.jsonl",
                              help="Path to save error cases")

    # --- compare ---
    compare_parser = subparsers.add_parser("compare", help="Compare two result files")
    compare_parser.add_argument("--a", required=True, help="Path to first results.json")
    compare_parser.add_argument("--b", required=True, help="Path to second results.json")
    compare_parser.add_argument("--label-a", default="Baseline", help="Label for first")
    compare_parser.add_argument("--label-b", default="Finetuned", help="Label for second")

    args = parser.parse_args()

    if args.command == "run":
        run_predictions(
            dataset_path=args.dataset,
            output_path=args.output,
            backend=args.backend,
            model=args.model,
            provider=args.provider,
            max_examples=args.max_examples,
            question_types=args.types,
        )

    elif args.command == "score":
        results = score_all(args.predictions)
        print_report(results)
        save_results(results, args.results)
        save_errors(results, args.errors)

    elif args.command == "compare":
        compare_results(args.a, args.b, args.label_a, args.label_b)


if __name__ == "__main__":
    main()
