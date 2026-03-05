"""Pipeline orchestrator for mechanical plate compliance dataset.

Generates a balanced dataset of PNG technical drawings paired with a JSONL
file containing specification text, questions, answers, reasoning, and
metadata. Distribution across rule complexities, annotation levels, and
violation counts is controlled via a JSON config file.

Usage:
    python orchestrator.py
    python orchestrator.py --config rl_config.json
    python orchestrator.py --num 500
"""

import argparse
import json
import os
import time
from typing import Any

import numpy as np

from question_generator import generate_questions
from renderer import decide_annotations, render_plate
from sampler import sample_plate_with_retry
from spec_generator import generate_spec

DEFAULT_CONFIG: dict[str, Any] = {
    "dataset": {"num_examples": 200, "seed": 0},
    "distribution": {
        "complexity_weights": {"simple": 0.3, "multi_rule": 0.3, "conditional": 0.4},
        "annotation_weights": {"full": 0.33, "partial": 0.33, "minimal": 0.34},
        "violation_counts": [0, 1, 2, 3],
    },
    "violations": {
        "allow_multi_violation": False,
        "spacing_oversample_weight": 2.0,
    },
    "sampling": {
        "max_retries": 30,
        "max_placement_attempts": 400,
    },
    "output": {"directory": "./dataset"},
}


def load_config(path: str | None) -> dict[str, Any]:
    """Load config from a JSON file, falling back to defaults.

    Performs a shallow merge: each top-level key in the user config
    updates the corresponding default dict.

    Args:
        path: Path to a JSON config file, or None for pure defaults.

    Returns:
        Merged configuration dictionary.
    """
    config = DEFAULT_CONFIG.copy()
    if path and os.path.exists(path):
        with open(path) as f:
            user_config = json.load(f)
        for key in user_config:
            if key in config and isinstance(config[key], dict):
                config[key].update(user_config[key])
            else:
                config[key] = user_config[key]
    return config


def build_schedule(
    num_examples: int, config: dict[str, Any]
) -> list[tuple[str, str, int]]:
    """Build a weighted schedule of (complexity, annotation_level, num_violations).

    Draws each axis independently from the configured weight distributions.

    Args:
        num_examples: Number of schedule entries to generate.
        config: Full configuration dictionary.

    Returns:
        List of (complexity, annotation_level, num_violations) tuples.
    """
    dist = config["distribution"]
    rng = np.random.default_rng(config["dataset"]["seed"])

    comp_names = list(dist["complexity_weights"].keys())
    comp_weights = np.array(list(dist["complexity_weights"].values()))
    comp_weights = comp_weights / comp_weights.sum()

    annot_names = list(dist["annotation_weights"].keys())
    annot_weights = np.array(list(dist["annotation_weights"].values()))
    annot_weights = annot_weights / annot_weights.sum()

    viol_counts = dist["violation_counts"]

    schedule: list[tuple[str, str, int]] = []
    for _ in range(num_examples):
        comp = rng.choice(comp_names, p=comp_weights)
        annot = rng.choice(annot_names, p=annot_weights)
        nv = int(rng.choice(viol_counts))
        schedule.append((comp, annot, nv))

    return schedule


def _convert_params(params: dict[str, Any]) -> dict[str, Any]:
    """Convert numpy types in rule params to native Python for JSON serialization.

    Args:
        params: Rule parameter dictionary potentially containing numpy scalars.

    Returns:
        Dictionary with all values as native Python types.
    """
    converted: dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, dict):
            converted[k] = {
                str(kk): float(vv) if isinstance(vv, (int, float)) else vv
                for kk, vv in v.items()
            }
        elif isinstance(v, (int, float)):
            converted[k] = float(v)
        else:
            converted[k] = v
    return converted


def generate_dataset(
    num_examples: int,
    output_dir: str,
    start_seed: int = 0,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate the full dataset: images, JSONL records, and statistics.

    For each scheduled example the pipeline samples a plate configuration,
    computes annotation visibility, generates specification text and
    questions, renders the technical drawing, and writes a JSONL record.

    Args:
        num_examples: Number of examples to generate.
        output_dir: Root output directory (images/ and dataset.jsonl created inside).
        start_seed: Starting seed for reproducibility.
        config: Configuration dictionary from load_config. Uses DEFAULT_CONFIG if None.

    Returns:
        Statistics dictionary summarising the generated dataset.
    """
    if config is None:
        config = DEFAULT_CONFIG

    allow_multi = config.get("violations", {}).get("allow_multi_violation", False)
    max_retries = config.get("sampling", {}).get("max_retries", 30)
    max_placement = config.get("sampling", {}).get("max_placement_attempts", 400)

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    jsonl_path = os.path.join(output_dir, "dataset.jsonl")
    schedule = build_schedule(num_examples, config)

    successes = 0
    failures = 0
    complexities = list(config["distribution"]["complexity_weights"].keys())
    annotations = list(config["distribution"]["annotation_weights"].keys())
    violation_counts = config["distribution"]["violation_counts"]
    stats: dict[str, Any] = {
        "complexity": {c: 0 for c in complexities},
        "annotation": {a: 0 for a in annotations},
        "violations": {v: 0 for v in violation_counts},
        "total_questions": 0,
        "question_types": {},
    }

    start_time = time.time()

    with open(jsonl_path, "w") as f:
        for i, (comp, annot, nv) in enumerate(schedule):
            seed = start_seed + i
            example_id = f"EX-{i:04d}"

            cfg = sample_plate_with_retry(
                num_violations=nv,
                rule_complexity=comp,
                annotation_level=annot,
                seed=seed,
                allow_multi_violation=allow_multi,
                max_retries=max_retries,
                max_placement_attempts=max_placement,
            )

            if cfg is None:
                failures += 1
                continue

            cfg.annotation_level = annot

            annot_rng = np.random.default_rng(seed)
            annot_vis = decide_annotations(cfg, annot_rng)

            spec_text = generate_spec(cfg, seed=seed)
            questions = generate_questions(cfg, seed=seed, annotations=annot_vis)

            image_filename = f"{example_id}.png"
            image_path = os.path.join(images_dir, image_filename)
            render_plate(cfg, image_path, seed=seed, annotations=annot_vis)

            holes_meta: list[dict[str, Any]] = []
            for h in cfg.holes:
                holes_meta.append(
                    {
                        "id": h.id,
                        "cx": h.cx,
                        "cy": h.cy,
                        "diameter": h.diameter,
                        "has_bolt": h.has_bolt,
                        "zone": h.zone,
                        "intended_violations": h.intended_violations,
                    }
                )

            rules_meta: list[dict[str, Any]] = []
            for r in cfg.rules:
                rules_meta.append(
                    {
                        "id": r.id,
                        "rule_type": r.rule_type,
                        "text": r.text,
                        "params": _convert_params(r.params),
                    }
                )

            record: dict[str, Any] = {
                "example_id": example_id,
                "image": f"images/{image_filename}",
                "spec_text": spec_text,
                "questions": questions,
                "metadata": {
                    "seed": seed,
                    "rule_complexity": comp,
                    "annotation_level": annot,
                    "num_violations": nv,
                    "plate_width": cfg.plate_width,
                    "plate_height": cfg.plate_height,
                    "num_holes": len(cfg.holes),
                    "num_rules": len(cfg.rules),
                    "zones": {k: v for k, v in cfg.zones.items()},
                    "holes": holes_meta,
                    "rules": rules_meta,
                },
            }

            f.write(json.dumps(record) + "\n")

            successes += 1
            stats["complexity"][comp] += 1
            stats["annotation"][annot] += 1
            stats["violations"][nv] += 1
            stats["total_questions"] += len(questions)
            for q in questions:
                qtype = q["type"]
                stats["question_types"][qtype] = (
                    stats["question_types"].get(qtype, 0) + 1
                )

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(
                    f"  [{i + 1}/{num_examples}] {rate:.1f} examples/sec "
                    f"({successes} ok, {failures} failed)"
                )

    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print("Dataset generation complete")
    print(f"{'=' * 60}")
    print(f"Output: {output_dir}")
    print(f"Time: {elapsed:.1f}s ({successes / elapsed:.1f} examples/sec)")
    print(
        f"Success: {successes}/{num_examples} ({successes / num_examples * 100:.0f}%)"
    )
    print(f"Failures: {failures}")
    print(f"\nDistribution:")
    print(f"  Complexity:  {stats['complexity']}")
    print(f"  Annotation:  {stats['annotation']}")
    print(f"  Violations:  {stats['violations']}")
    print(f"\nQuestions:")
    print(f"  Total: {stats['total_questions']}")
    print(f"  By type: {stats['question_types']}")
    print(f"  Avg per example: {stats['total_questions'] / max(successes, 1):.1f}")

    stats_path = os.path.join(output_dir, "stats.json")
    stats["successes"] = successes
    stats["failures"] = failures
    stats["num_examples"] = num_examples
    stats["elapsed_seconds"] = round(elapsed, 1)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Generate VLM compliance dataset")
    parser.add_argument(
        "--num", type=int, default=None, help="Number of examples (overrides config)"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Starting seed (overrides config)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(SCRIPT_DIR, "config.json"),
        help="Config file path",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Config loaded from: {args.config} (exists: {os.path.exists(args.config)})")
    print(f"Generating {config['dataset']['num_examples']} examples")

    num = args.num or config["dataset"]["num_examples"]
    output = args.output or config["output"]["directory"]
    seed = args.seed if args.seed is not None else config["dataset"]["seed"]

    os.makedirs(output, exist_ok=True)
    config_save_path = os.path.join(output, "config.json")
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=2)

    generate_dataset(
        num_examples=num,
        output_dir=output,
        start_seed=seed,
        config=config,
    )
