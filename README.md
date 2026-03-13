# Multimodal Reasoning on Synthetic Data

**Author:** Joshua Kames-King

---

A synthetic data generation and finetuning pipeline for improving vision-language model performance on engineering design compliance checking. The project demonstrates that SFT on targeted synthetic data produces genuine but shallow cross-modal reasoning: Ministral 3 8B goes from 51.8% to 87.5% balanced accuracy on compliance checking, with measurement extraction error dropping from 8.14mm to 2.84mm. Ablation studies confirm the model uses both image and specification inputs, but reveal that its visual reasoning is limited to label reading and coarse spatial estimation.

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

| Metric | Baseline (197 ex.) | Unbalanced SFT (30 ex.) | Balanced SFT (50 ex.) |
|--------|----------|----------------|--------------|
| Overall accuracy | 13.5% | 93.3% | 91.1% |
| Yes accuracy | 5.3% | 98.7% | 92.1% |
| No accuracy | 98.3% | 47.2% | 82.8% |
| **Balanced accuracy** | **51.8%** | **73.0%** | **87.5%** |
| Measurement MAE | 8.14mm | 3.25mm | 2.84mm |
| Counterfactual MAE | 4.85mm | 0.17mm | 0.07mm |
| Rule selection | 0% | 100% | 100% |
| Audit F1 | 0.228 | 0.467 | 0.501 |

Balanced accuracy (average of Yes and No accuracy) is the primary metric because the test set is ~90% Yes. A model that always says "Yes" achieves 90% raw accuracy but only 50% balanced accuracy. No accuracy (violation detection) is the most operationally important metric: in a compliance checking system, the entire value is in catching violations.

### Baseline: Ministral 8B (197 test examples, 3,418 compliance questions)

The base model exhibits extreme "No" bias, predicting non-compliance for virtually everything (98.3% No accuracy, 5.3% Yes accuracy). Despite this, it extracts measurements reasonably well (diameter MAE 0.42mm), suggesting the bottleneck is reasoning, not vision.

**3x3 Grid (annotation x complexity):**

|           | Simple | Multi-rule | Conditional |
|-----------|--------|------------|-------------|
| Full      | 19.7%  | 14.2%      | 12.0%       |
| Partial   | 20.8%  | 13.5%      | 11.8%       |
| Minimal   | 16.7%  | 9.8%       | 8.9%        |

Both difficulty gradients are visible: simple -> conditional (18.8% -> 11.1%) and full -> minimal (14.7% -> 11.6%).

### Finetuned: Balanced SFT (50 test examples, 857 compliance questions)

Trained on 250 examples with 30% No oversampling using LoRA (rank 64, alpha 128) on Ministral 3 8B-Instruct with FP8 weights. LoRA targets both language model and vision encoder layers (they share the same projection names). Training: 1 epoch, ~4 hours on A100 80GB.

**3x3 Grid (annotation x complexity):**

|           | Simple | Multi-rule | Conditional |
|-----------|--------|------------|-------------|
| Full      | 92.0% (n=75)  | 93.2% (n=88)  | 94.1% (n=220) |
| Partial   | 77.8% (n=45)  | 88.3% (n=60)  | 91.0% (n=188) |
| Minimal   | 90.6% (n=117) | 75.0% (n=16)  | 95.8% (n=48)  |

The annotation gradient partially holds (full > partial consistently), but the complexity gradient is inverted (conditional scores highest). See the Analysis section for an explanation.

### Measurement Improvement

| Type | Baseline | Finetuned | Improvement |
|------|----------|-----------|-------------|
| Diameter | 0.42mm | 0.13mm | 3x better |
| Edge distance | 11.99mm | 1.91mm | 6x better |
| Hole-to-hole | 20.10mm | 9.33mm | 2x better |
| Plate dimensions | 0.0mm | 0.0mm | (perfect) |

Spatial measurements (edge distance, hole-to-hole) improved substantially but hole-to-hole remains the hardest at 9.33mm MAE. The measurement faithfulness analysis below explains why.

### Measurement MAE by Annotation Level

|           | Full  | Partial | Minimal |
|-----------|-------|---------|---------|
| Overall   | 1.69mm | 3.84mm | 3.81mm |

The annotation gradient is clearly visible in measurements: the model is measurably worse at extracting numbers when labels are absent. This gradient is hidden in the compliance metric because the binary Yes/No decision is tolerant of measurement error -- you can get the measurement somewhat wrong and still get the compliance answer right, especially when 90% of answers are Yes.

## Ablation Studies

We ran two ablation experiments on 30 test examples each, using the same random seed as the main evaluation for comparability.

### Setup

**No-image ablation:** Replace the technical drawing with a blank white image. If compliance accuracy remains high, the model is using a text-only shortcut.

**No-spec ablation:** Replace the specification document with a placeholder string. If accuracy remains high, the model extracts everything from the image alone.

### Results: Both Modalities Required

|                     | Full model | No image | No spec |
|---------------------|------------|----------|---------|
| Overall accuracy    | 91.1%      | 77.1%    | 88.2%   |
| Yes accuracy        | 92.1%      | 81.6%    | 93.7%   |
| No accuracy         | 82.8%      | 37.7%    | 41.5%   |
| **Balanced accuracy** | **87.5%** | **59.7%** | **67.6%** |

The model requires both modalities for violation detection. No accuracy collapses from 82.8% to 37.7% without images and to 41.5% without the specification. This rules out H2 (text-only shortcut) and confirms genuine cross-modal reasoning.

### Violation Detection by Annotation Level

Breaking down No accuracy by annotation level reveals which images the model depends on:

|           | Full model | No image | Drop |
|-----------|------------|----------|------|
| Full      | 84.2% (n=38) | 15.8% (n=19) | -68.4 |
| Partial   | 84.6% (n=26) | 36.8% (n=19) | -47.8 |
| Minimal   | 78.3% (n=23) | 66.7% (n=15) | -11.6 |

The pattern is the opposite of what spatial reasoning would predict. At full annotation, the model is completely dependent on the image for violation detection (68 point drop when removed), because it reads dimension labels. At minimal annotation, removing the image barely matters (11.6 point drop), because the model is largely solving these from text patterns in the spec and question. The model has learned to read labels, not to reason spatially.

Note: sample sizes are small (15-38 per cell). The direction of the gradient across all three levels (68 > 48 > 12) is consistent.

## Analysis: What Did the Model Learn?

### Measurement Faithfulness

The model claims "Using the scale bar" in its reasoning chains for minimal annotation examples. To test whether this is genuine, we compared measurements with and without the image at minimal annotation:

| Type | MAE with image | MAE no image | MAE no spec | Interpretation |
|------|----------------|--------------|-------------|----------------|
| Plate dims | 0.0mm | 25.3mm | 0.0mm | **Genuine visual reasoning** from image only |
| Hole-to-hole | 8.5mm | 40.9mm | 8.1mm | **Partial visual reasoning**, imprecise |
| Diameter | 0.4mm | 0.9mm | 1.7mm | **Mostly spec priors** (nominal values in rules) |
| Edge distance | 6.5mm | 5.1mm | 8.0mm | **No reliable strategy** |

The image is essential for large-scale features (plate outline, relative hole positions) but not for small-scale measurements (diameter, edge distance). The model says "Using the scale bar" even when looking at a blank white image, confirming the chain-of-thought text is a learned template, not faithful reasoning.

Critically, even though the reasoning text is unfaithful, the actual numerical outputs change with the image for plate dims and hole-to-hole, confirming that genuine visual processing occurs at the representation level. The faithfulness gap is in the verbalized reasoning, not the underlying computation.

### Adaptive Strategy by Annotation Level

At full annotation, the model reads labels directly:

> "The annotation on H1 reads diameter 6.2mm."

> "H3 is at (102.1, 67.4) on a 149.0x92.0mm plate. Distances to edges: left=102.1mm, right=46.9mm, bottom=67.4mm, top=24.6mm."

At minimal annotation, the model switches to estimation:

> "Using the scale bar, H6 appears approximately 10mm in diameter."

> "Using the scale bar, the plate appears approximately 131x87mm."

The model adapts its strategy to the annotation level. At full it reads labels; at minimal it estimates. But the quality of the estimation varies drastically by measurement type: perfect for plate dimensions, poor for spatial relationships between components.

### Training Data Drives the Strategy

At full annotation, the training data provides reasoning chains with exact ground truth coordinates:

> "H4 is at (43.2, 18.7) on a 90.0x75.0mm plate. Distances to edges: left=43.2mm, right=46.8mm, bottom=18.7mm, top=56.3mm."

At minimal annotation, training chains use vague estimation:

> "Using the scale bar, H4 appears approximately 22mm from the bottom edge."

These coordinates are not visible in the images. They come from the data generation pipeline. At inference, the model hallucinates coordinates in the learned format and does arithmetic on them. Sometimes this is accurate (when the model can roughly perceive positions), sometimes it is not.

This reveals a fundamental property of SFT: the model learns to use the easiest information pathway available in the training data. For fully annotated images, that pathway is label reading (OCR + arithmetic). The model never needed to develop precise visual measurement because the labels provided exact values. Visual reasoning only emerges as a fallback for features where no textual shortcut exists.

### Text Priors Enable Shortcut at Minimal Annotation

The specification text provides strong priors that reduce the need for visual measurement:

- **Nominal values**: "Nominal diameter for Zone A holes: 8.0 mm" lets the model guess diameters without measuring
- **Threshold values**: "Minimum hole-to-hole spacing shall be 35.0 mm" gives the decision boundary
- **Training distribution**: ~90% of holes comply, so guessing "Yes" without measuring is statistically reliable

At minimal annotation, the model's effective strategy is: "I know the threshold from the spec, I know most holes comply from training, I'll say Yes unless something in the image is obviously wrong." This achieves ~91% compliance accuracy without precise spatial measurement.

### Why the Difficulty Gradient Disappears

The baseline shows both gradients (annotation: full 14.7% -> minimal 11.6%; complexity: simple 18.8% -> conditional 11.1%). After finetuning, these collapse in the compliance metric.

**Complexity gradient (inverted):** The training data explicitly teaches the multi-hop lookup chain for conditional rules. Once the model learns the template ("read spec -> find threshold -> read value -> compare"), conditional rules are not harder than simple ones. The training data eliminates the reasoning difficulty by demonstration. This supports H1 (template matching).

**Annotation gradient (hidden):** The gradient IS present in measurement MAE (full 1.69mm vs minimal 3.81mm), proving the model is worse at extracting numbers without labels. But the compliance metric hides this because the binary Yes/No threshold is forgiving: you can be off by several mm and still get the compliance decision right, especially when 90% of answers are Yes.

### Hypotheses Resolved

The original hypotheses for the missing gradient:

- **H1 (Template matching): Confirmed.** The model learned a single reasoning template that handles all complexity levels. The training data teaches the lookup chain explicitly.
- **H2 (Text-only shortcut): Rejected.** The ablation shows removing images drops balanced accuracy by 27.8 points. The model genuinely uses images.
- **H3 (Saturated difficulty): Partially confirmed.** The clean matplotlib drawings are easy enough that the model can read labels with high accuracy regardless of annotation level. The model's visual reasoning is real but limited to coarse spatial features.

The full picture is a combination: the model uses genuine cross-modal reasoning (rejecting H2), but the reasoning is shallow -- it consists of label reading at full annotation and coarse spatial estimation at minimal, rather than precise spatial measurement. SFT teaches the easiest pathway present in the training data.

## Known Limitations and Next Steps

### Current Limitations

- **Yes/No imbalance**: Test set is ~90% Yes. Per-cell No counts are too small (0-38) for reliable grid analysis of violation detection.
- **50-example evaluation**: Balanced SFT evaluated on 50 of 197 test examples. Extending to 197 would tighten grid statistics.
- **Unfaithful chain-of-thought**: The model verbalizes "Using the scale bar" even without an image. The reasoning text is a learned template, not an accurate description of the model's process.
- **Training data leaks coordinates**: Full annotation reasoning chains contain ground truth coordinates from the data pipeline, teaching coordinate hallucination rather than visual extraction.
- **Synthetic-to-real gap**: Matplotlib drawings with clean geometry are far simpler than real CAD output. Transfer to real engineering documents (DesignQA) is untested.

### Next Experiments

- **DesignQA transfer**: Evaluate on real CAD drawings to test whether the learned reasoning transfers beyond synthetic data. If it fails, this confirms that the model's visual skills are specific to clean synthetic images.
- **Vision degradation check**: Run ChartQA/DocVQA before and after finetuning to verify that domain-specific training did not degrade general vision capabilities.
- **Hard-mode synthetic data**: Generate training images with labels stripped even at "full" annotation, forcing the model to develop genuine visual measurement strategies rather than label reading.
- **Explicit spatial reasoning chains**: Replace coordinate-based chains with scale bar reasoning ("the scale bar shows 20mm spans approximately 150 pixels; the hole diameter spans 47 pixels; therefore diameter ~ 6.3mm"). This would teach a transferable visual measurement strategy.
- **RL with GRPO**: Use compliance accuracy as reward to let the model discover its own visual reasoning strategies. SFT is limited to imitating demonstrated reasoning, but RL can optimize for correct answers regardless of the reasoning path. This is most promising for spatial measurement where the correct reasoning approach is not obvious.
- **Curriculum on annotation stripping**: Start training with fully labeled images (teaching the reasoning template), then progressively remove labels (forcing visual strategies). The model builds on the template while developing visual measurement skills.
- **Per-cell oversampling**: Balance training data within each grid cell independently, particularly increasing No examples to improve violation detection.

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

Supports both API-based evaluation (Mistral, OpenAI, Anthropic) and local inference with LoRA adapters. Ablation modes (`--ablation no-image` and `--ablation no-spec`) enable controlled information removal for diagnosing what the model has learned.

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

**Annotation-aware reasoning chains** ensure training data matches what the model can see. Full annotation chains cite exact values; minimal annotation chains reference the scale bar and approximate values. However, we discovered that full annotation chains leak ground truth coordinates from the data pipeline, which may teach coordinate hallucination rather than genuine visual extraction (see Analysis section).

## Development

### Code Quality

- **Language:** Python 3.12
- **Type hints:** All functions have comprehensive type annotations
- **Dependencies:** numpy, matplotlib, torch, transformers, peft, trl, bitsandbytes

### GPU Requirements

- Training and evaluation require A100 80GB (or A800 80GB)
- Model loads with `FineGrainedFP8Config()` which auto-dequantizes to bf16 on compute capability 8.0
- **Do not use Blackwell/RTX PRO GPUs**: FP8 stays native on compute capability 9.0+, causing LoRA math to fail
- **Do not strip FP8 config with `delattr`**: produces garbage weights
- Batch size 1, gradient accumulation 8, max sequence length 4096
