# Qwen LoRA/SFT Repair Result

Run source: Kaggle kernel `kostyaryazanov/stage3-qwen25vl-lora-sft-repair-clean`, version 1.

This run repaired the previous LoRA smoke enough to train, pass the overfit gate, and complete full clean `val_v2` evaluation. The output JSON fields are valid, but the learned decision policy is not useful as a Stage 3 candidate.

## Setup

- Base model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Adaptation: LoRA on clean `train_balanced_v2`
- Training samples: 96 stratified samples
- Prompt policy: no `crop_path`, no class-like filename text
- Eval split: clean `val_v2`, 58 GT crops
- Overfit gate: 6 train examples before full validation

## Overfit Gate

| check | result |
|---|---:|
| parse OK | 6/6 |
| coarse OK | 4/6 |
| full val attempted | yes |

The repaired recipe fixed the earlier punctuation/invalid-output failure. It did not fix the semantic bias problem.

## Corrected Full-Val Metrics

The original Kaggle artifact omitted `parse_status` and `schema_valid` in `sample_results.jsonl`, which made evaluator parse/schema counters read as zero. The predictions file itself passes `validate_vlm_labels_v1.py`. A corrected local evaluation sets parse/schema from the parsed records.

| metric | value |
|---|---:|
| parse success | 1.0000 |
| schema valid | 1.0000 |
| coarse accuracy | 0.5172 |
| coarse macro-F1 | 0.1579 |
| visibility accuracy | 0.7931 |
| visibility macro-F1 | 0.2949 |
| tag mean Jaccard | 0.3261 |
| pred ambiguous rate | 0.0000 |

Confusion matrix:

```text
gt\pred          insulator_ok  defect_flashover  defect_broken  unknown
insulator_ok               29                 1              2        0
defect_flashover           16                 1              3        0
defect_broken               6                 0              0        0
```

## Interpretation

This is a negative adaptation result. The model learned stable JSON and more normal-looking tags, but it collapsed toward `insulator_ok`. It solved neither `defect_flashover` recall nor `defect_broken` recall. The high raw accuracy is not useful because it is driven by the majority normal class.

## Decision

Do not promote this LoRA adapter to Stage 4. Keep it as evidence that a naive small LoRA/SFT recipe can learn format while worsening the class boundary. If LoRA is revisited, it needs a stronger class-balanced objective or a discriminative auxiliary loss, not just another small wording or learning-rate tweak.

## Artifacts

- Raw Kaggle archive: `outputs/_kaggle_downloads/stage3_qwen25vl_lora_sft_repair_clean_v1/stage3_deliverables_qwen25vl_3b_lora_sft_repair_clean.tar.gz`
- Corrected local eval: `outputs/_external_runs/stage3_qwen25vl_lora_sft_repair_clean_corrected/eval_corrected/metrics.json`
