# Indeed Policy Eval Harness — Implementation Guide

## Overview

This eval harness measures how well Phoebe's LLM-as-judge safety classifier detects Indeed job posting policy violations. It maps 10 Indeed content policies to GA Guard categories and computes F1, precision, recall, and confusion matrices.

### Expected performance

| System | F1 | Notes |
|---|---|---|
| GA Guard (adversarial RL) | 0.840 | Per-policy fine-tuned compact models |
| GPT-4o baseline | 0.623 | Zero-shot LLM-as-judge |
| **Phoebe (this harness)** | **~0.60-0.65** | LLM-as-judge via Claude, no fine-tuning |

Phoebe trades per-classification cost for zero-retraining flexibility — it evaluates any policy without retraining, which is the fundamental tradeoff vs GA's approach.

## Quick Start

### 1. Download the dataset

```bash
# Create the data directory
mkdir -p data/indeed

# Download from Kaggle (requires kaggle CLI + API key)
kaggle datasets download -d arshkon/linkedin-job-postings -p data/indeed/ --unzip
# OR for the Indeed-specific dataset:
kaggle datasets download -d promptcloud/indeed-job-posting-dataset -p data/indeed/ --unzip

# Rename to expected filename
mv data/indeed/*.csv data/indeed/job_postings.csv
```

**Important:** The raw Kaggle dataset has no ground-truth violation labels. You need to add a `ground_truth` column. See [Labeling](#labeling-the-dataset) below.

### 2. Generate synthetic violations

The Kaggle dataset is >92% compliant. Without synthetic violations, you'll get meaningless recall numbers (near 0 because there are almost no violations to detect).

```bash
# Generate 200 synthetic violations per policy (2,000 total)
python main.py generate-synthetic --n-per-policy 200

# Or generate alongside eval
python main.py eval --generate-synthetic 200
```

### 3. Run the eval

```bash
# Full eval (all 10 policies, all samples)
python main.py eval --model-api-key sk-ant-...

# Sample 100 examples for a quick test
python main.py eval --sample 100 --model-api-key sk-ant-...

# Eval specific policies only
python main.py eval --policies discrimination,pii_harvesting,scam_postings

# Write results to JSON
python main.py eval --output results/indeed_eval.json

# Higher concurrency (faster, more API cost)
python main.py eval --concurrency 10
```

## Architecture

```
src/eval/
├── __init__.py
├── indeed_policies.py   # 10 Indeed policies → GA Guard category mapping
├── dataset.py           # CSV loader, sampler, stratified sampling
├── metrics.py           # ConfusionMatrix, EvalResults, F1/precision/recall
├── runner.py            # Orchestrates classification runs
└── synthetic.py         # Template-based synthetic violation generator
```

### Flow

```
1. Load CSV from data/indeed/
2. (Optional) Load synthetic_violations.csv
3. For each sample × each policy:
   a. Build classification prompt from Indeed policy description
   b. Map to GA Guard category
   c. Call SafetyClassifier.classify()
   d. Compare predicted_unsafe vs ground_truth
4. Compute per-policy and overall metrics
5. Print summary + write JSON
```

### Indeed → GA Guard Policy Mapping

| Indeed Policy | GA Guard Category | What it catches |
|---|---|---|
| Discrimination | `hate` | Protected class bias in hiring |
| Misleading Information | `misinformation` | Fake salary, bait-and-switch |
| Illegal Activities | `illicit_activities` | Drug trafficking, fraud recruitment |
| PII Harvesting | `pii_ip` | SSN/bank details in applications |
| Sexual Content | `sexual_content` | Adult services, explicit language |
| Violence & Threats | `violence_self_harm` | Physical intimidation in job ads |
| Scam Postings | `illicit_activities` + `misinformation` | Advance-fee, pyramid schemes |
| Hate Speech | `hate` | Slurs, derogatory language |
| Privacy Violations | `pii_ip` + `prompt_security` | Forced credential sharing |
| Child Labor & Safety | `violence_self_harm` + `illicit_activities` | Minor labor law violations |

## Labeling the Dataset

The raw Kaggle CSV needs a `ground_truth` column. Options:

### Option A: Manual labeling (recommended for small sets)

Add a column to your CSV:

```csv
job_text,ground_truth
"Great software engineer role at Acme Corp...",compliant
"Young attractive females only, apply now!",discrimination
"Send SSN and bank details to apply","pii_harvesting,scam_postings"
```

Values: comma-separated Indeed policy IDs, or `compliant` for clean postings.

### Option B: Use synthetic violations only

Skip labeling the Kaggle data entirely and evaluate only against synthetic violations:

```bash
python main.py generate-synthetic --n-per-policy 200
python main.py eval --data-dir data/indeed
```

This tests classifier accuracy on known violations but doesn't measure false positive rate on real-world compliant postings.

### Option C: LLM-assisted labeling

Use Phoebe itself to pre-label, then manually review:

```bash
# Use the T&S dashboard to classify individual postings
python main.py arena
# Then POST to /api/tns/classify with each posting
```

## Synthetic Augmentation

### Template-based (built-in)

```bash
python main.py generate-synthetic --n-per-policy 200
```

Generates deterministic violations from templates with randomized fill values. Good for regression testing but won't catch edge cases.

### Red-team generated (recommended)

Use Phoebe's red team mode to generate adversarial violations that are harder to detect:

```bash
python main.py redteam --auto
```

This generates LLM-crafted attacks that test the classifier's robustness against evasion techniques (paraphrasing, obfuscation, context manipulation).

Save the outputs to `data/indeed/synthetic_violations.csv` with the same schema.

### Getting GA's synthetic set

GA supplemented their eval with ~2K synthetic violations using adversarial RL. If you have access to their dataset, place it at `data/indeed/synthetic_violations.csv`.

## Interpreting Results

### Output format

```
======================================================================
EVAL RESULTS — Indeed Policy Classification
======================================================================
Total samples:         2100
Total classifications: 21000

Overall F1:        0.6234
Overall Precision: 0.7012
Overall Recall:    0.5612

Macro F1:          0.5987

----------------------------------------------------------------------
Policy                    F1      Prec    Recall    TP    FP    TN    FN
----------------------------------------------------------------------
discrimination         0.6500   0.7200  0.5900    59    23   877    41
misleading_info        0.5800   0.6100  0.5500    55    35   865    45
...
----------------------------------------------------------------------

Comparison benchmarks:
  GA Guard (adversarial RL):  F1 0.8400
  GPT-4o baseline:            F1 0.6230
  Phoebe (this run):          F1 0.6234
```

### What to look for

- **High FP (false positives):** Classifier is too aggressive — over-flagging compliant postings. Common with broad GA Guard categories like `hate`.
- **High FN (false negatives):** Classifier misses violations — the main gap vs GA Guard. Expected for subtle/obfuscated violations.
- **Per-policy variance:** Some policies (e.g., `pii_harvesting`) are easier to detect than others (e.g., `misleading_info` with subtle bait-and-switch).

### To improve F1 toward GA's 0.840

1. **Fine-tune per-policy classifiers** — this is what GA's adversarial RL pipeline does
2. **Add Indeed-specific few-shot examples** to the classifier prompt
3. **Use the red team mode** to find classifier blind spots and add them to training
4. **Ensemble multiple models** — run both Claude and a fine-tuned compact model

## T&S Dashboard Integration

All classified breaches from `POST /api/tns/classify` land in the breach log. Run the arena server alongside the eval to see results in the dashboard:

```bash
# Terminal 1: start arena
python main.py arena

# Terminal 2: run eval (it calls the classifier directly, not the API)
python main.py eval --output results/eval_run_1.json
```

The eval harness calls `SafetyClassifier.classify()` directly (not through the HTTP API), so results won't appear in the T&S dashboard by default. To log to the dashboard, use the `--via-api` flag (not yet implemented — PRs welcome).
