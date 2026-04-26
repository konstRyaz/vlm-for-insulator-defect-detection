# Stage 4 Metric Improvement Plan

This note starts from the clean v7f Stage 4 run, not from the historical
crop-path-leakage runs.

## Current Diagnosis

| Item | Value |
| --- | ---: |
| Stage 3 GT-crop ceiling | 0.4655 |
| Stage 4 actual pipeline correct rate | 0.3621 |
| Ceiling minus actual gap | 0.1034 |
| Detector match rate | 1.0000 |
| Good crop rate among matched | 0.9828 |
| VLM correct rate on good predicted crops | 0.3684 |

The detector is not the main bottleneck on this validation split. The main
loss is coarse-class interpretation by the VLM on geometrically good crops.

## Focused Error Review

A targeted review package was generated from the clean v7f Stage 4 artifacts:

- HTML: `outputs/_external_runs/stage4_clean_v7f_with_ceiling/04_eval/targeted_review/targeted_error_review.html`
- Markdown: `outputs/_external_runs/stage4_clean_v7f_with_ceiling/04_eval/targeted_review/targeted_error_review.md`
- Case table: `outputs/_external_runs/stage4_clean_v7f_with_ceiling/04_eval/targeted_review/targeted_error_review.csv`

Main coarse error patterns:

| Pattern | Count |
| --- | ---: |
| `insulator_ok -> defect_flashover` | 15 |
| `insulator_ok -> defect_broken` | 7 |
| `defect_flashover -> defect_broken` | 4 |
| `defect_flashover -> insulator_ok` | 3 |
| `defect_broken -> defect_flashover` | 3 |
| `defect_flashover -> unknown` | 2 |
| `defect_broken -> unknown` | 1 |
| `insulator_ok -> unknown` | 1 |

The most important failure is overcalling defects on normal insulators,
especially `defect_flashover`.

## Next Best Experiment

The next experiment should test input context rather than another broad prompt
sweep.

Recommended comparison:

| Variant | Description | Reason |
| --- | --- | --- |
| tight crop | current baseline | reference |
| context crop | same predicted/GT box with larger padding | tests whether missing local context causes false flashover |
| two-view crop | tight crop plus context crop, if the runner can support it minimally | tests whether the VLM needs both detail and context |

The first runnable ablation is:

- notebook: `notebooks/stage4_context_pad050_kaggle_run.ipynb`
- run name: `stage4_detector_to_vlm_pred_val_context_pad050_kaggle`
- crop padding: `0.50`
- prompt: `qwen_vlm_labels_v1_prompt_v7f_flashover_unclear_to_unknown_nocroppath`

Acceptance signal:

- `insulator_ok -> defect_flashover` should decrease.
- Stage 3 ceiling and Stage 4 actual should improve together.
- Detector match/good-crop rates should remain unchanged for Stage 4.
- Parse/schema validity must remain 1.0.

If context does not help, the likely next step is no longer prompt tuning. It
would be either a rule-assisted VLM baseline or supervised fine-tuning, both of
which should be reported as separate methods rather than as the frozen Stage 3
baseline.
