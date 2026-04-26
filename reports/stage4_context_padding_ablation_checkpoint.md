# Stage 4 Context Padding Ablation Checkpoint

This checkpoint records the first context-crop ablation after the clean v7f
Stage 4 baseline.

## Runs Compared

| Run | Crop padding | Qwen max_pixels | Status |
| --- | ---: | ---: | --- |
| tight clean baseline | 0.15 | default | valid |
| context pad050 first run | 0.50 | default | not final: 21 Qwen OOM backend errors |

The first context run is useful diagnostically, but it is not a final comparable
result because 21 predicted crops failed during Qwen inference.

## Corrected Ceiling Comparison

The Kaggle notebook did not have the clean Stage 3 ceiling attached, so the
ceiling was recomputed locally from the clean v7f Stage 3 run.

| Metric | tight 0.15 | context 0.50 uncapped |
| --- | ---: | ---: |
| detector match rate | 1.0000 | 1.0000 |
| good crop rate among matched | 0.9828 | 0.9828 |
| VLM correct rate on good pred crops | 0.3684 | 0.3860 |
| pipeline correct rate | 0.3621 | 0.3793 |
| Stage 3 ceiling correct rate | 0.4655 | 0.4655 |
| ceiling minus actual gap | 0.1034 | 0.0862 |
| Qwen successful pred-crop outputs | 214/215 | 194/215 |

## Main Error Shift

| Error pattern | tight 0.15 | context 0.50 uncapped |
| --- | ---: | ---: |
| `insulator_ok -> defect_flashover` | 15 | 10 |
| `insulator_ok -> defect_broken` | 7 | 5 |
| `defect_flashover -> insulator_ok` | 3 | 7 |
| `defect_flashover -> defect_broken` | 4 | 5 |
| `defect_broken -> defect_flashover` | 3 | 2 |

Interpretation: wider context reduces overcalling defects on normal insulators,
which supports the input-context hypothesis. However, it also weakens flashover
recall and caused memory failures on T4.

## Next Rerun

The notebook and config were updated for a comparable rerun:

- notebook: `notebooks/stage4_context_pad050_kaggle_run.ipynb`
- run name: `stage4_detector_to_vlm_pred_val_context_pad050_maxpix401k_kaggle`
- crop padding: `0.50`
- Qwen visual cap: `max_pixels=401408`

Acceptance signal for the capped rerun:

- no Qwen backend OOM errors;
- parse/schema stability restored;
- `insulator_ok -> defect_flashover` remains below the tight baseline;
- pipeline correct rate improves or at least the trade-off is interpretable.

