# Next research results

This folder closes the next research block requested after the Stage 3/4 checkpoint. The goal was to compare direct VLM, no-VLM visual-feature classifiers and hybrid classifier+VLM reporter systems under the clean no-crop-path protocol.

## Completed blocks

- `protocol/` - train/validation protocol and overlap audit.
- `non_vlm_baselines/` - DINOv2/CLIP/SigLIP/timm feature classifiers with train-CV selection.
- `structured_output_eval/` - automatic and draft rubric evaluation of VLM JSON/report fields.
- `vlm_benefit/` - comparison of class accuracy versus structured-reporting value.
- `accuracy_ablation/flashover_binary/` - targeted `insulator_ok` vs `defect_flashover` diagnostic.

## Main numbers

| system | role | result |
|---|---|---:|
| DINOv2-base + LogisticRegression | best no-VLM class-only baseline | 0.6552 acc / 0.6684 macro-F1 |
| Qwen Stage 3 clean | direct VLM structured reporter | 0.4655 acc |
| Qwen Stage 4 context | direct VLM detector-to-crop baseline | 0.3966 pipeline correct |
| DINOv2 + Qwen champion | hybrid classifier+reporter | 0.5862 pipeline correct |

## Interpretation

The no-VLM DINOv2 classifier is the strongest raw crop-level classifier in this checkpoint. That does not remove the VLM module: it changes its role. Qwen is best justified as a structured reporter that produces JSON fields, tags, visibility/review signals and human-readable snippets. The strongest architecture is therefore hybrid: discriminative visual features for `coarse_class`, VLM for report structure.

Manual visual review is still recommended before making strong claims about description quality. The current filled rubric is a draft structured-field pass, not a final expert visual annotation.
