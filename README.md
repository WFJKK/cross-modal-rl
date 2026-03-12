# Multimodal Reasoning on Synthetic Data

**Author:** Joshua Kames-King

---

A synthetic data generation and finetuning pipeline for improving vision-language model performance on engineering design compliance checking. The project demonstrates that SFT on targeted synthetic data can dramatically improve cross-modal reasoning: Ministral 3 8B goes from 51.8% to 84.9% balanced accuracy on compliance checking, with measurement extraction error dropping from 8.14mm to 3.17mm.

The pipeline generates technical drawings of mechanical plates with holes, paired with specification documents and exhaustive question-answer sets with step-by-step reasoning chains. Two controllable difficulty axes (annotation density and rule complexity) enable both curriculum training and diagnostic evaluation.

## Motivation

Current VLMs struggle with cross-modal reasoning on engineering documents, requiring models to extract rules from text specifications and apply them to visual diagrams. DesignQA showed that even GPT-4o and LLaVA perform poorly on compliance checking tasks.

It is a priori not obvious what the exact failure mode is: a lack of ability of VLMs to infer quantitative data from the images, or multi-modal reasoning itself, or even a combination of both. There are hints in the literature that VLMs struggle significantly with quantitative visual questions ("how far apart are these objects?"), see for example Liao et al. (Q-Spatial, arXiv:2409.09788). Hence, we consider a self-created synthetic dataset that targets both failure modes through two controllable meta-parameters:

1. **Annotation density**: Each drawing is generated at three levels: fully annotated (all dimensions labeled), partially annotated (some dimensions removed), and unannotated (no dimension labels). At full annotation the task reduces to OCR + logic, isolating the reasoning component. At no annotation the model must infer measurements from visual proportions alone, directly training the quantitative spatial skill that the literature identifies as deficient. SpatialVLM (Chen et al., CVPR 2024) showed that this gap is data-driven rather than architectural, so we are hopeful that synthetic data will help here too.

2. **Rule complexity**: Specification documents range from simple single-threshold rules ("all holes shall have diameter 8.0 +/- 0.3mm") to conditional rules that require multi-hop cross-modal reasoning ("for Class A joints where hole spacing < 20mm, minimum edge distance shall be >= 2.0x hole diameter"). This forces the model to perform image -> text -> image -> compute chains of increasing depth, targeting the cross-modal reasoning gap identified by DesignQA.

By varying these two axes independently, the dataset serves both as training data (graduated difficulty acts as a curriculum) and as a diagnostic tool (performance across the 3x3 grid reveals whether failures stem from spatial inference, reasoning, or their combination).

Each example requires the model to:
1. **Parse rules** from a specification document (text)
2. **Extract measurements** from a technical drawing (image)
3. **Apply rules to measurements** and determine compliance (reasoning)

## Results

### Summary

| Metric | Baseline | Unbalanced SFT | Balanced SFT |
|--------|----------|----------------|--------------|
| Overall accuracy | 13.5% | 93.3% | 89.4% |
| Yes accuracy | 5.3% | 98.7% | 90.6% |
| No accuracy | 98.3% | 47.2% | 79.2% |
| **Balanced accuracy** | **51.8%** | **73.0%** | **84.9%** |
| Measurement MAE | 8.14mm | 3.25mm | 3.17mm |
| Counterfactual MAE | 4.85mm | 0.17mm | 0.09mm |
| Rule selection | 0% | 100% | 100% |
| Audit F1 | 0.228 | 0.467 | 0.493 |

Balanced accuracy (average of Yes and No accuracy) is the primary metric because the test set is ~90% Yes. A model that always says "Yes" achieves 90% raw accuracy but only 50% balanced accuracy.

### Baseline: Ministral 8B (197 test examples, 3,418 compliance questions)

The base model exhibits extreme "No" bias, predicting non-compliance for virtually everything (98.3% No accuracy, 5.3% Yes accuracy). Despite this, it extracts measurements reasonably well (diameter MAE 0.42mm), suggesting the bottleneck is reasoning, not vision.

**3x3 Grid (annotation x complexity):**

|           | Simple | Multi-rule | Conditional |
|-----------|--------|------------|-------------|
| Full      | 19.7%  | 14.2%      | 12.0%       |
| Partial   | 20.8%  | 13.5%      | 11.8%       |
| Minimal   | 16.7%  | 9.8%       | 8.9%        |

Both difficulty gradients are visible: simple -> conditional (18.8% -> 11.1%) and full -> minimal (14.7% -> 11.6%).

### Finetuned: Balanced SFT (30 test examples, 510 compliance questions)

Trained on 250 examples with 30% No oversampling using LoRA (rank 64, alpha 128) on Ministral 3 8B-Instruct with FP8 weights. LoRA targets both language model and vision encoder layers (they share the same projection names). Training: 1 epoch, ~4 hours on A100 80GB.

**3x3 Grid (annotation x complexity):**

|           | Simple | Multi-rule | Conditional |
|-----------|--------|------------|-------------|
| Full      | 88.9%  | 83.3%      | 92.2%       |
| Partial   | 66.7%  | 88.3%      | 91.9%       |
| Minimal   | 90.3%  | 75.0%      | 100.0%      |

Note: 30 test examples produce uneven cell sizes (12-124 questions) with very few No examples per cell (0-12). The expected difficulty gradients are not reliably visible at this sample size. Full 197-example evaluation is pending.

**Balanced accuracy per cell (avg of Yes acc and No acc):**

|           | Simple    | Multi-rule | Conditional |
|-----------|-----------|------------|-------------|
| Full      | 81.8% (7N) | 90.0% (2N) | 86.7% (10N) |
| Partial   | 52.4% (3N) | 93.6% (5N) | 91.5% (11N) |
| Minimal   | 80.2% (12N) | 75.0% (0N) | 100.0% (3N) |

Cells marked with N count show ground-truth No examples. Cells with <5 No examples have unreliable balanced accuracy.

### Measurement Improvement

| Type | Baseline | Finetuned | Improvement |
|------|----------|-----------|-------------|
| Diameter | 0.42mm | 0.14mm | 3x better |
| Edge distance | 11.99mm | 2.50mm | 5x better |
| Hole-to-hole | 20.10mm | 10.05mm | 2x better |
| Plate dimensions | 0.0mm | 0.0mm | (perfect) |

Spatial measurements (edge distance, hole-to-hole) improved substantially but hole-to-hole remains the hardest at 10mm MAE.

## Dataset Architecture

The pipeline has five stages:

```
sampler.py -> spec_generator.py -> question_generator.py -> renderer.py -> orchestrator.py
```

### 1. Parameter Sampler (`sampler.py`)

Generates plate configurations with controlled compliance states using **constructive sampling**: decides the desired outcome first (which rules should be violated, by which holes), places holes to achieve that exact compliance state, and verifies geometric validity and single-rule violations. This avoids the reject-and-retry problem of random generation.

### 2. Specification Generator (`spec_generator.py`)

Converts plate configurations into readable specification documents with varying complexity: simple (2-3 rules, direct statements), multi-rule (4 rules including bolt population), and conditional (4 rules, two zones, table lookups, material-class mapping requiring multi-hop reasoning).

### 3. Question Generator (`question_generator.py`)

Produces exhaustive question-answer pairs with annotation-aware reasoning chains: per-component compliance (every hole x every rule), full audit, measurement extraction, rule selection (conditional only), and counterfactual reasoning.

### 4. Image Renderer (`renderer.py`)

Generates technical drawings at three annotation levels: full (all dimensions labeled), partial (some annotations hidden), and minimal (hole IDs and scale bar only).

### 5. Pipeline Orchestrator (`orchestrator.py`)

Generates balanced datasets with even distribution across 3 rule complexities x 3 annotation levels x 4 violation counts = 36 combinations.

## Training Pipeline

### Data Loader (`training/data_loader.py`)

Loads dataset JSONL, splits train/val by example (not by question, to prevent image leakage), flattens records into individual question-answer pairs, formats as Mistral multimodal chat conversations, and provides a custom collator for batching images and text with proper label masking.

### Training Script (`training/train_ministral8b.py`)

Fine-tunes Ministral 3 8B with LoRA adapters using HuggingFace TRL. The model loads with FP8 weights (auto-dequantizes to bf16 on A100 GPUs). Key configuration: LoRA rank 64 targeting all attention and MLP layers in both language model and vision encoder, 8-bit AdamW optimizer, cosine learning rate schedule with warmup, gradient checkpointing.

```bash
# Training
cd training
python train_ministral8b.py \
    --dataset ../data/train/dataset.jsonl \
    --output ../results/finetuned/ministral-8b-lora-balanced \
    --oversample 0.3

# Evaluation (requires GPU for local inference)
python evaluate.py run \
    --model results/finetuned/ministral-8b-lora-balanced \
    --backend local \
    --dataset data/test/dataset.jsonl \
    --output results/finetuned/ministral-8b-lora-balanced/predictions.jsonl

# Score and compare
python evaluate.py score --predictions results/finetuned/ministral-8b-lora-balanced/predictions.jsonl \
    --results results/finetuned/ministral-8b-lora-balanced/results.json
python evaluate.py compare \
    --a results/baseline/ministral-8b/results.json \
    --b results/finetuned/ministral-8b-lora-balanced/results.json \
    --label-a Baseline --label-b "Balanced SFT"
```

## Evaluation

The evaluation script (`evaluate.py`) scores model predictions on five metrics:

1. **Compliance accuracy** (overall + 3x3 annotation x complexity grid + balanced accuracy)
2. **Measurement MAE** in mm (overall + by annotation level + by measurement type)
3. **Audit F1** (precision, recall, F1 for violation detection)
4. **Counterfactual MAE** (backward reasoning: "what value would make this comply?")
5. **Rule selection accuracy** (spec parsing for conditional complexity)

Supports both API-based evaluation (Mistral, OpenAI, Anthropic) and local inference with LoRA adapters.

## Usage

### Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### Generate Dataset

```bash
uv run python orchestrator.py                              # Default: 200 examples
uv run python orchestrator.py --num 500 --output ./my_data # Custom
```

### Dataset Structure

```
data/
  train/
    dataset.jsonl   # 2,769 training examples
    images/         # PNG technical drawings
  test/
    dataset.jsonl   # 197 test examples
    images/
```

## Design Decisions

**Constructive sampling** guarantees exact control over the compliance state. Random placement leads to either mostly compliant or ambiguous multi-rule violations.

**Three annotation levels** create a natural curriculum: full annotation teaches the reasoning pattern (OCR + logic), minimal annotation forces visual inference, and the same questions apply regardless.

**Exhaustive hole x rule questions** teach systematic compliance checking. A model trained on incomplete checks would learn to be incomplete.

**Annotation-aware reasoning chains** ensure training data matches what the model can see. Full annotation chains cite exact values; minimal annotation chains reference the scale bar and approximate values.

## Open Question: Where Is the Difficulty Gradient?

The dataset was designed with two independent difficulty axes: annotation density (full -> partial -> minimal) controls how much visual information is available, and rule complexity (simple -> multi_rule -> conditional) controls reasoning depth. In the baseline, both gradients are visible (simple 18.8% -> conditional 11.1%, full 14.7% -> minimal 11.6%).

After finetuning, these gradients disappear. All cells in the 3x3 grid score 75-100% with no clear pattern. This raises a critical question: **did the model learn genuine cross-modal reasoning, or did it learn a shortcut?**

Three hypotheses:

**H1: Template matching.** The model learned a single reasoning template ("read spec -> find number -> compare -> answer") that works equally well across all cells. Once you learn the template, conditional rules aren't harder than simple ones because the training data explicitly teaches the multi-hop lookup chain.

**H2: Text-only shortcut.** The model may be answering correctly without using the image at all. If the specification text and question contain enough information to guess the answer (e.g., the question mentions "H1" and "Rule R1", and the spec defines R1's tolerance), the image becomes redundant. This would explain why annotation level doesn't matter — the model never looks at annotations.

**H3: Saturated difficulty.** The synthetic images (clean matplotlib drawings, perfect circles, consistent scale bars) may be too easy even at "minimal" annotation. Real CAD drawings have clutter, overlapping annotations, 3D projections, and ambiguous dimensions that would restore the gradient.

### Planned Ablation Studies

To distinguish these hypotheses, we test the finetuned model under controlled information removal:

**No-image ablation:** Replace the technical drawing with a blank white image. If compliance accuracy remains high, the model is using a text-only shortcut (supports H2). If accuracy drops significantly, the model genuinely uses visual information.

**No-spec ablation:** Replace the specification document with a placeholder. If accuracy remains high, the model extracts everything from the image alone. If accuracy drops, the model needs the spec to know which rules to check.

| No-image result | No-spec result | Interpretation |
|-----------------|----------------|----------------|
| High | Low | Text shortcut: model ignores images |
| Low | High | Visual shortcut: model ignores specs |
| Low | Low | Genuine cross-modal reasoning |
| High | High | Memorized answers: neither input needed |

A secondary test is **DesignQA transfer**: running the finetuned model on real CAD drawings from the DesignQA benchmark. If synthetic-to-real transfer fails, this supports H3 (synthetic images are too easy) and motivates harder synthetic data generation or domain adaptation.

These ablations are more informative than additional training runs because they diagnose *what* the model learned rather than *how well* it performs.

## Known Limitations and Next Steps

### Current Limitations

- **Yes/No imbalance**: Test set is ~90% Yes. Balanced accuracy (84.9%) is the reliable metric, not raw accuracy (89.4%). Per-cell analysis requires more No examples for reliable grid statistics.
- **Synthetic-to-real gap**: Matplotlib drawings, not real CAD output. Transfer to real engineering documents (e.g. DesignQA) is untested.
- **30-example evaluation**: Current finetuned results use 30 test examples. Full 197-example evaluation is in progress.
- **No ablation studies yet**: It is unclear whether the model uses the image, the specification text, or both for its predictions.

### Planned Experiments

- **Full evaluation**: Run all 197 test examples for reliable per-cell statistics
- **Ablation studies**: Test model performance without image (text-only) and without specification (image-only) to determine what the model actually learned
- **DesignQA transfer**: Evaluate on real CAD drawings to test synthetic-to-real transfer
- **Vision benchmarks**: Run ChartQA/DocVQA before and after finetuning to check for capability degradation
- **Per-cell oversampling**: Balance training data within each grid cell independently
- **RL finetuning**: Use GRPO with compliance accuracy as reward to push beyond SFT ceiling, particularly for spatial measurement extraction where correct reasoning paths cannot be demonstrated

## Development

### Code Quality

- **Language:** Python 3.12
- **Type hints:** All functions have comprehensive type annotations
- **Dependencies:** numpy, matplotlib, torch, transformers, peft, trl, bitsandbytes
