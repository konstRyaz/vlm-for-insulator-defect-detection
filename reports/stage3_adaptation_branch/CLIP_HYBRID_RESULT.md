# Stage 3 CLIP / Hybrid Coarse Benchmark Result

Run source: Kaggle kernel `kostyaryazanov/notebookd64e91cba0`, version 22.

This experiment tested a coarse-only CLIP-style branch for the hybrid idea:

`crop -> discriminative coarse classifier -> Qwen structured reporter`

It does not replace the `vlm_labels_v1` structured-output path. It only checks whether visual embeddings can improve the difficult `insulator_ok` versus `defect_flashover` boundary.

## Result

| method | accuracy | macro_f1_3class | macro_f1_with_unknown | ok_recall | flashover_recall | broken_recall |
| --- | --- | --- | --- | --- | --- | --- |
| clip_zero_shot | 0.3103 | 0.2858 | 0.1715 | 0.3438 | 0.2000 | 0.5000 |
| clip_linear_probe | 0.4483 | 0.3148 | 0.1889 | 0.3750 | 0.7000 | 0.0000 |
| clip_linear_probe_unknown_threshold_045 | 0.2069 | 0.1739 | 0.1043 | 0.0000 | 0.6000 | 0.0000 |

## Interpretation

The zero-shot CLIP prompt ranking is too weak for direct use. The linear probe is more interesting: it reaches `0.4483` accuracy and `0.3148` three-class macro-F1, with `0.70` recall on `defect_flashover`. That is close to the clean Qwen2.5-VL-3B accuracy range and slightly higher in macro-F1 than the latest clean Qwen control rerun, but it loses the `defect_broken` class completely.

This is not a final replacement for Qwen. It is a useful signal that a small discriminative component can recover flashover cases, but it needs either more data, better features, or class-specific handling for `defect_broken` before it can become a Stage 3/4 candidate.

## Decision

Continue with the planned Qwen2.5-VL-3B LoRA/QLoRA smoke run. Keep the CLIP branch as a promising auxiliary direction, not as the main result yet.
