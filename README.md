# Synthetic Dataset Generation for VLM Cross-Modal Reasoning

**Author:** Joshua Kames-King

---

A synthetic data generation pipeline that produces multimodal training data for vision-language models on engineering design compliance checking. The pipeline generates technical drawings of mechanical plates with holes, paired with specification documents and exhaustive question-answer sets with step-by-step reasoning chains.

This dataset aims to specifically train VLMs to 1) strengthen quantitative visual "understanding" and 2) perform multi-modal multi-hop reasoning as described in more detail in the next subsection.

## Motivation and General Idea

Current VLMs struggle with cross-modal reasoning on engineering documents — extracting rules from text specifications and applying them to visual diagrams. DesignQA showed that even GPT-4o and LLaVA perform poorly on compliance checking tasks.

It is a priori not obvious what the exact failure mode is: a lack of ability of VLMs to infer quantitative data from the images, or multi-modal reasoning itself, or even a combination of both. There are hints in the literature that VLMs struggle significantly with quantitative visual questions ("how far apart are these objects?"), see for example Liao et al. (Q-Spatial, arXiv:2409.09788). Hence, we consider a self-created synthetic dataset that targets both failure modes through two controllable meta-parameters:

1. **Annotation density**: Each drawing is generated at three levels: fully annotated (all dimensions labeled), partially annotated (some dimensions removed), and unannotated (no dimension labels). At full annotation the task reduces to OCR + logic, isolating the reasoning component. At no annotation the model must infer measurements from visual proportions alone, directly training the quantitative spatial skill that the literature identifies as deficient. SpatialVLM (Chen et al., CVPR 2024) showed that this gap is data-driven rather than architectural — training on synthetic spatial data significantly improved quantitative estimation. Hence we are hopeful that synthetic data will help here too.

2. **Rule complexity**: Specification documents range from simple single-threshold rules ("all holes shall have diameter 8.0 ± 0.3mm") to conditional rules that require multi-hop cross-modal reasoning ("for Class A joints where hole spacing < 20mm, minimum edge distance shall be ≥ 2.0× hole diameter"). This forces the model to perform image → text → image → compute chains of increasing depth, targeting the cross-modal reasoning gap identified by DesignQA.

By varying these two axes independently, the dataset serves both as training data (graduated difficulty acts as a curriculum) and as a diagnostic tool (performance across the grid reveals whether failures stem from spatial inference, reasoning, or their combination).

In addition, we include step-by-step reasoning chains as worked examples in the training data (but not as part of the evaluation).

Each example requires the model to:
1. **Parse rules** from a specification document (text)
2. **Extract measurements** from a technical drawing (image)
3. **Apply rules to measurements** and determine compliance (reasoning)

## Project Structure
```
VLMRLSFT/
├── sampler.py               # Parameter sampler (constructive placement)
├── spec_generator.py        # Specification document generator
├── question_generator.py    # Q/A pair + reasoning chain generator
├── renderer.py              # Technical drawing renderer (matplotlib)
├── orchestrator.py          # Pipeline orchestrator (dataset generation)
├── evaluate.py              # Evaluation: run predictions, score, compare
├── config.json              # Default orchestrator config
├── config_train.json        # Training dataset config (3000 examples, seed 0)
├── config_test.json         # Test dataset config (200 examples, seed 9999)
├── training/
│   ├── data_loader.py       # Data loading, splitting, formatting for SFT
│   └── trainministral8b.py  # LoRA fine-tuning script (Ministral 3 8B)
├── data/
│   ├── train/               # Generated training data
│   │   ├── dataset.jsonl
│   │   ├── stats.json
│   │   └── images/
│   └── test/                # Generated test/eval data
│       ├── dataset.jsonl
│       ├── stats.json
│       └── images/
└── results/
    └── finetuned/
        └── ministral-8b-lora/  # Saved LoRA adapter weights
```

## Architecture

The pipeline has five stages:
```
sampler.py → spec_generator.py → question_generator.py → renderer.py → orchestrator.py
```

### 1. Parameter Sampler (`sampler.py`)

Generates plate configurations with controlled compliance states using **constructive sampling**:
- Decides the desired outcome first (which rules should be violated, by which holes)
- Places holes to achieve that exact compliance state
- Verifies geometric validity and single-rule violations

This avoids the reject-and-retry problem of random generation. Each violating hole fails exactly one intended rule, giving clean ground truth.

**Parameters:**
- Plate dimensions: 80–160mm × 50–100mm
- Holes: 3–8 per plate, diameters 6–16mm
- Four rule types: tolerance, edge distance, spacing, bolt population

### 2. Specification Generator (`spec_generator.py`)

Converts plate configurations into readable specification documents. The same underlying rules are presented with varying complexity:

- **Simple**: 2–3 rules, direct statements ("All holes shall have diameter 10.0 ± 0.5mm")
- **Multi-rule**: 4 rules including bolt population, single zone
- **Conditional**: 4 rules, two zones, table lookups, material-class mapping requiring multi-hop reasoning

Rule order is shuffled in the document while preserving IDs, forcing models to locate rules by ID rather than position.

For conditional complexity, the model must chain multiple lookups:
1. Read material from header → "Aluminum 6061-T6"
2. Look up material-to-class mapping → Class II
3. Find tolerance table, Class II row
4. Determine hole size category (small/large)
5. Extract tolerance and compute acceptable range

### 3. Question Generator (`question_generator.py`)

Produces exhaustive question-answer pairs with **annotation-aware reasoning chains**:

- **Per-component compliance**: Every hole × every rule checked individually
- **Full audit**: "List all violations" — tests systematic completeness
- **Measurement extraction**: Distance between holes or edge distances
- **Rule selection** (conditional only): Tests spec parsing ("What tolerance applies to Zone B?")
- **Counterfactual**: "What minimum edge distance would H3 need to comply?" — tests backward reasoning

The reasoning chains adapt to the annotation level:
- **Full**: Exact values from labels — "H1 diameter is 7.9mm."
- **Partial**: Mix of exact (for visible labels) and approximate — "From the scale bar, H2 diameter appears approximately 8mm."
- **Minimal**: All approximate — "From the scale bar, H3 appears approximately 10mm from the top edge."

This ensures the training data teaches reasoning that matches what the model can actually see in each image. The annotation visibility decisions are shared between the renderer and question generator so they always agree on what's shown.

### 4. Image Renderer (`renderer.py`)

Generates technical drawings with three annotation levels:

- **Full**: All diameters labeled, edge distances shown, spacing annotations
- **Partial**: Some annotations randomly hidden (ensuring hidden ones are relevant to violations)
- **Minimal**: Hole IDs only, scale bar, no dimension labels

The annotation level controls how hard it is to extract measurements from the image. The same questions apply regardless of annotation level — only the visual information changes.

### 5. Pipeline Orchestrator (`orchestrator.py`)

Generates balanced datasets with weighted distribution across:
- 3 rule complexities × 3 annotation levels × 4 violation counts

Distribution weights are configured via JSON config files. Outputs per dataset:
- `dataset.jsonl` — one record per example with image path, spec text, all QA pairs, and full metadata
- `images/` — PNG technical drawings
- `stats.json` — dataset statistics

## Usage

### Setup
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### Generate Datasets

The pipeline uses JSON config files to control dataset size, distribution weights, and output location. Two configs are provided:
```bash
# Generate training set (3000 examples → data/train/)
uv run python orchestrator.py --config config_train.json

# Generate test set (200 examples → data/test/)
uv run python orchestrator.py --config config_test.json

# Default config (1000 examples → dataset/)
uv run python orchestrator.py

# Override with custom config
uv run python orchestrator.py --config my_config.json
```

**Config format** (see `config_train.json`, `config_test.json`):
```json
{
  "dataset": { "num_examples": 3000, "seed": 0 },
  "distribution": {
    "complexity_weights": { "simple": 0.3, "multi_rule": 0.3, "conditional": 0.4 },
    "annotation_weights": { "full": 0.33, "partial": 0.33, "minimal": 0.34 },
    "violation_counts": [1, 2, 3, 4]
  },
  "violations": { "allow_multi_violation": false, "spacing_oversample_weight": 2.0 },
  "sampling": { "max_retries": 30, "max_placement_attempts": 400 },
  "output": { "directory": "./data/train" }
}
```

The training config uses `violation_counts: [1, 2, 3, 4]` (every example has at least one violation), while the test config uses `[0, 1, 2, 3]` (includes fully compliant examples for realistic evaluation).

### Output Structure
```
data/train/                    data/test/
├── dataset.jsonl              ├── dataset.jsonl
├── stats.json                 ├── stats.json
└── images/                    └── images/
    ├── EX-0000.png                ├── EX-0000.png
    ├── EX-0001.png                ├── EX-0001.png
    └── ...                        └── ...
```

### JSONL Record Format
```json
{
  "example_id": "EX-0000",
  "image": "images/EX-0000.png",
  "spec_text": "SPEC-GP-672: Guide Plate GP-672 Design Requirements\n\nRule R1: ...",
  "questions": [
    {
      "type": "per_component_compliance",
      "question": "Does hole H1 comply with Rule R1?",
      "answer": "Yes",
      "reasoning": "H1 diameter is 8.1mm. Rule R1 specifies nominal 8.0 ± 0.3mm..."
    }
  ],
  "metadata": {
    "seed": 0,
    "rule_complexity": "simple",
    "annotation_level": "full",
    "num_violations": 1,
    "plate_width": 131.0,
    "plate_height": 63.0,
    "holes": [...],
    "rules": [...]
  }
}
```

## Evaluation

The evaluation script (`evaluate.py`) handles prediction generation, scoring, and model comparison via three subcommands.

### Running Predictions
```bash
# Run a model against the test set
uv run python evaluate.py run \
    --model mistral-small-latest \
    --dataset data/test/dataset.jsonl \
    --output results/predictions_baseline.jsonl \
    --provider mistral

# Supported providers: mistral, openai, anthropic
# Use --backend local for HuggingFace models (e.g. finetuned adapters)
uv run python evaluate.py run \
    --model results/finetuned/ministral-8b-lora \
    --dataset data/test/dataset.jsonl \
    --output results/predictions_finetuned.jsonl \
    --backend local

# Filter to specific question types
uv run python evaluate.py run \
    --model mistral-small-latest \
    --dataset data/test/dataset.jsonl \
    --types per_component_compliance full_audit \
    --max-examples 50
```

Predictions support resume — rerunning the same command skips already-completed questions.

### Scoring
```bash
# Score predictions against ground truth
uv run python evaluate.py score \
    --predictions results/predictions_baseline.jsonl \
    --results results/results_baseline.json \
    --errors results/errors_baseline.jsonl
```

This produces:
- `results.json` — structured scores across all metrics and breakdowns
- `errors.jsonl` — every incorrect prediction with expected vs actual for error analysis
- Console report with all metrics

### Comparing Models
```bash
# Compare baseline vs finetuned
uv run python evaluate.py compare \
    --a results/results_baseline.json \
    --b results/results_finetuned.json \
    --label-a "Baseline" \
    --label-b "Finetuned"
```

Prints a side-by-side comparison with deltas across all metrics, including the 3×3 annotation × complexity grid.

### Metrics

1. **Compliance classification accuracy**: Binary Yes/No for each hole × rule pair, broken down by annotation level (full → partial → minimal gap measures spatial inference ability), rule complexity (simple → conditional gap measures multi-hop reasoning), rule type, and answer balance (Yes vs No bias detection).

2. **Full audit F1**: Precision catches hallucinated violations, recall catches missed ones.

3. **Measurement extraction MAE**: Absolute error in mm between predicted and true distances. Directly measures spatial information extraction.

4. **Rule understanding accuracy**: Can the model correctly parse conditional specs to find applicable parameters?

5. **Counterfactual MAE**: Can the model compute correct thresholds for compliance? Tests backward reasoning.

The 3×3 grid (annotation level × rule complexity) is the key diagnostic: it decomposes model failures into spatial inference deficits (column differences) versus reasoning deficits (row differences).

## Baseline Results

We evaluated two Mistral models on the generated test set using `evaluate.py`:

### Pixtral Large (30 examples, 681 predictions)

**Compliance accuracy: 135/488 = 27.7%**

| Annotation | Accuracy | | Complexity  | Accuracy |
|------------|----------|-|-------------|----------|
| Full       | 40/133 = 30.1% | | Simple      | 34/126 = 27.0% |
| Partial    | 50/197 = 25.4% | | Multi-rule  | 39/170 = 22.9% |
| Minimal    | 45/158 = 28.5% | | Conditional | 62/192 = 32.3% |

**Answer bias:** Yes 23.2%, No 67.3% — strong "No" bias. The model defaults to predicting non-compliance, inflating its score on actual violations but missing most compliant cases.

**3×3 grid (annotation × complexity):**

|           | Simple | Multi-rule | Conditional |
|-----------|--------|------------|-------------|
| Full      | 6/36 = 16.7% | 15/48 = 31.3% | 19/49 = 38.8% |
| Partial   | 10/30 = 33.3% | 14/80 = 17.5% | 26/87 = 29.9% |
| Minimal   | 18/60 = 30.0% | 10/42 = 23.8% | 17/56 = 30.4% |

**Other metrics:**
- Measurement MAE: 28.2mm overall (diameter 2.2mm, edge distance 23.4mm, hole-to-hole 41.9mm, plate dims 45.5mm)
- Audit F1: 0.259 (precision 26.4%, recall 58.6%)
- Counterfactual MAE: 7.4mm
- Rule selection: 0/6

### Ministral 3 8B (3 examples, 69 predictions)

**Compliance accuracy: 4/50 = 8.0%**

**Answer bias:** Yes 0.0%, No 100.0% — extreme "No" bias. The model predicts non-compliance for virtually every question.

**Other metrics:**
- Measurement MAE: 6.2mm overall (diameter 0.03mm, edge distance 13.1mm, hole-to-hole 11.7mm, plate dims 0.0mm)
- Audit F1: 0.274 (precision 21.4%, recall 88.9%)
- Counterfactual MAE: 10.4mm

### Interpretation

Both models perform well below chance on the balanced binary compliance task, confirming that this dataset presents a genuine challenge for current VLMs.

The dominant failure mode is a strong "No" bias — both models default to predicting non-compliance. This is particularly severe for Ministral 3 8B (8% accuracy, answering "No" to nearly everything), while Pixtral Large shows a milder version of the same pattern (27.7% overall). This bias likely stems from the models' tendency to err on the side of caution when uncertain about compliance.

Pixtral Large's compliance breakdown shows relatively flat performance across annotation levels (25–30%) and complexity levels (23–32%), suggesting the model fails at a fundamental level before the difficulty gradient becomes the bottleneck. The flat annotation axis (no clear drop from full → minimal) implies the model cannot reliably extract measurements even when they are explicitly labeled in the image.

Measurement extraction errors confirm this: while diameter readings from labeled annotations are reasonable (2.2mm MAE), spatial distances like hole-to-hole (41.9mm MAE) and plate dimensions (45.5mm MAE) are far off, indicating the models struggle with visual spatial reasoning even for basic geometric properties.

These baselines establish the pre-training performance floor that fine-tuning aims to improve upon.

## Training

### Data Loading (`training/data_loader.py`)

Prepares the generated dataset for SFT training:
- Splits at the record level (all questions from one plate stay together, preventing data leakage)
- Flattens records into individual examples (one per question)
- Oversamples minority "No" compliance answers to balance the ~90/10 Yes/No imbalance
- Formats as chat messages matching the evaluation prompt structure

### Fine-tuning (`training/trainministral8b.py`)

LoRA fine-tuning of Ministral 3 8B with HuggingFace TRL:
```bash
cd training

# Default settings (3 epochs, LoRA r=64, 4-bit quantization)
python trainministral8b.py \
    --dataset ../data/train/dataset.jsonl \
    --output ../results/finetuned/ministral-8b-lora

# Custom hyperparameters
python trainministral8b.py \
    --dataset ../data/train/dataset.jsonl \
    --output ../results/finetuned/ministral-8b-lora \
    --epochs 5 --lr 1e-4 --batch-size 4 --lora-r 128 --lora-alpha 256

# Full precision (needs more VRAM)
python trainministral8b.py --no-4bit
```

**Requirements:**
```bash
pip install torch transformers peft trl bitsandbytes
pip install git+https://github.com/huggingface/transformers.git  # latest for Ministral 3
```

**Configuration:**
- LoRA targets: all attention projections + MLP layers (`q/k/v/o_proj`, `gate/up/down_proj`)
- Quantization: 4-bit NF4 with double quantization (fits in 24 GB VRAM)
- Optimiser: AdamW 8-bit with cosine LR schedule
- Gradient checkpointing enabled for memory efficiency

### End-to-End Workflow
```bash
# 1. Generate datasets
uv run python orchestrator.py --config config_train.json
uv run python orchestrator.py --config config_test.json

# 2. Baseline evaluation
uv run python evaluate.py run \
    --model pixtral-large-latest --provider mistral \
    --dataset data/test/dataset.jsonl \
    --output results/predictions_baseline.jsonl
uv run python evaluate.py score \
    --predictions results/predictions_baseline.jsonl \
    --results results/results_baseline.json

# 3. Fine-tune
cd training
python trainministral8b.py --dataset ../data/train/dataset.jsonl
cd ..

# 4. Finetuned evaluation
uv run python evaluate.py run \
    --model results/finetuned/ministral-8b-lora --backend local \
    --dataset data/test/dataset.jsonl \
    --output results/predictions_finetuned.jsonl
uv run python evaluate.py score \
    --predictions results/predictions_finetuned.jsonl \
    --results results/results_finetuned.json

# 5. Compare
uv run python evaluate.py compare \
    --a results/results_baseline.json \
    --b results/results_finetuned.json \
    --label-a "Pixtral Large" --label-b "Finetuned"
```

## Design Decisions

### Why constructive sampling?

Random hole placement with post-hoc compliance checking leads to either: (a) most examples being fully compliant (which is not useful), or (b) messy multi-rule violations that make ground truth ambiguous. Constructive sampling guarantees exact control over the compliance state.

### Why three annotation levels?

This creates a natural curriculum:
- **Full annotation** teaches the reasoning pattern: parse rule → extract value → compute → conclude
- **Minimal annotation** forces visual inference: the model already knows the reasoning, but must get numbers from geometry instead of labels, which might be one of the failure modes as described in the motivation section
- The same questions and answers apply regardless — only the information source changes

### Why exhaustive hole × rule questions?

Each example checks every hole against every rule. This teaches systematic compliance checking rather than cherry-picking. A model trained on incomplete checks would learn to be incomplete.

### Why separate train/test configs?

The training config (`config_train.json`) uses `violation_counts: [1, 2, 3, 4]` — every plate has at least one violation, maximising the density of informative examples. The test config (`config_test.json`) uses `[0, 1, 2, 3]` — including fully compliant plates for realistic evaluation. Different seeds (0 vs 9999) ensure no overlap between the two sets.

### Avoiding data contamination

The data teaches reasoning skills, not memorizable facts, by construction:
- Every example has unique plate dimensions, hole positions, and rule parameters
- Rule order is shuffled in specs
- Material-class mapping is consistent but tolerance values vary per example
- The model cannot memorize "Aluminum = ±0.5mm" because the tolerance for each class changes between examples

## Known Limitations & Future Work

- **Spacing violations are underrepresented** (~5% of violations) due to geometric constraints in constructive placement. Improved from <1% by biasing placement toward plate interior, but still lower than other rule types.
- **Visual fidelity is synthetic** — matplotlib drawings, not real CAD output. Transfer to real engineering documents is an open question.
- **Linguistic diversity is limited** — template-based specs use the same sentence patterns. An LLM paraphrase step could add variety.

## Development

### Package Management

Uses `uv` with `pyproject.toml` for dependency management.

### Code Quality

- **Language:** Python 3.12
- **Formatting:** Black (line length 88)
- **Type hints:** All functions have comprehensive type annotations
- **Docstrings:** Google-style on all public functions and classes
- **Type checking:** Passes `pyright` in basic mode with 0 errors, 0 warnings
- **Dependencies:** numpy, matplotlib (generation); torch, transformers, peft, trl (training)
```bash
# Verify type checking
uv run pyright sampler.py spec_generator.py question_generator.py renderer.py orchestrator.py evaluate.py

# Verify formatting
uv run black --check *.py training/*.py
```