# VLM Expansion Plan

This note defines the next research branch after the clean Stage 3/4 checkpoint.
The frozen Qwen2.5-VL-3B baseline remains the reference result; the experiments below are not replacements until they beat it under the same leakage-free protocol.

## Why This Branch Exists

The clean Stage 3/4 results show that the main bottleneck is VLM semantics, especially `insulator_ok` versus `defect_flashover`. Detector geometry is not the limiting factor on the current validation slice. That makes stronger VLMs and supervised adaptation reasonable next experiments.

## Guardrails

Keep the existing clean protocol unchanged:

- no `crop_path` or class-like filename tokens in prompts;
- same train/val split;
- same `vlm_labels_v1` output contract for generative VLMs;
- same Stage 3 GT-crop ceiling evaluation;
- same Stage 4 predicted-crop actual evaluation;
- every new model must be compared against the frozen Qwen v7f clean baseline.

Do not mix model families in one metric unless the output contract is comparable. A generative VLM can be evaluated on the full schema. A CLIP-style model can be evaluated on coarse labels and maybe ranking, but it is not automatically a full `vlm_labels_v1` producer.

## Candidate Families

### 1. Stronger General VLMs

First compare frozen inference before training anything.

Candidates:

- Qwen2.5-VL-7B-Instruct, if Kaggle/Colab memory permits;
- newer Qwen-VL family checkpoints, if available and compatible with the existing runner;
- other open VLMs only if they can run on the same cloud GPU budget and produce stable JSON.

Expected value: checks whether the current bottleneck is mostly model capacity rather than prompt wording.

### 2. Power-Domain Generative VLMs

Candidates mentioned for audit:

- PowerGPT;
- Power-LLaVA or similar power-line inspection MLLMs.

These should be tested only if weights and inference code are actually available. If only a paper exists, record it as related work rather than an executable baseline.

Expected value: domain pretraining may help flashover-like surface cues and reduce false positives on normal insulators.

### 3. TL-CLIP / CLIP-Style Models

TL-CLIP appears to be a power-specific contrastive vision-language model for transmission-line defect recognition. Treat it as a classifier or embedding/ranking baseline, not as a direct Stage 3 structured-output replacement.

Possible evaluations:

- zero-shot coarse label classification using text prompts;
- image-text similarity ranking over class descriptions;
- feature extraction followed by a small linear classifier trained only on the clean train split.

Expected value: may improve coarse classification, but it cannot natively produce `visual_evidence_tags`, `visibility`, or report snippets.

### 4. Fine-Tuning Current Qwen

Only after frozen-model comparisons.

Preferred first training path:

- LoRA/QLoRA on Qwen2.5-VL-3B-Instruct;
- train on clean Stage 3 train labels only;
- evaluate on clean `val_v2` only;
- preserve exact output schema;
- compare against frozen Qwen v7f under the same evaluator.

Expected value: most likely route to improve flashover/ok semantics, but also the easiest place to overfit because the dataset is small.

## Recommended Experiment Order

1. Availability audit: which models have public weights, runnable inference code, and a license compatible with this project.
2. Frozen Qwen scale test: Qwen2.5-VL-7B or the nearest runnable stronger Qwen checkpoint.
3. Frozen power-domain VLM test: PowerGPT/Power-LLaVA only if weights are usable.
4. TL-CLIP coarse-only classifier benchmark.
5. Qwen LoRA/QLoRA supervised fine-tune.
6. Optional CLIP-feature linear probe if TL-CLIP or a similar checkpoint is available.

## Minimal Metrics

For generative VLMs:

- parse success;
- schema valid;
- coarse accuracy;
- coarse macro-F1;
- visibility accuracy;
- visibility macro-F1;
- tag mean Jaccard;
- Stage 4 pipeline correct rate.

For CLIP-style models:

- coarse accuracy;
- coarse macro-F1;
- per-class recall;
- confusion matrix;
- optional top-k class ranking accuracy.

## Stop Rules

Stop a frozen VLM candidate if it cannot produce stable JSON or cannot run on the target cloud GPU without reducing the input so aggressively that the comparison becomes unfair.

Stop a fine-tune candidate if validation improves only by moving errors around within a tiny number of samples, or if it improves Stage 3 but hurts Stage 4 predicted crops.

## Current Research Question

Can a stronger or domain-adapted VLM reduce the `insulator_ok` versus `defect_flashover` confusion without relying on leaked path metadata and without degrading structured-output stability?
