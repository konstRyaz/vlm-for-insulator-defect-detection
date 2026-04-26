# Stage 4 Context Padding Ablation Checkpoint

This checkpoint records the first context-crop ablation after the clean v7f
Stage 4 baseline.

## Runs Compared

| Run | Crop padding | Qwen max_pixels | Status |
| --- | ---: | ---: | --- |
| tight clean baseline | 0.15 | default | valid |
| context pad050 first run | 0.50 | default | not final: 21 Qwen OOM backend errors |
| context pad030 capped run | 0.30 | 401408 | valid |
| context pad050 capped run | 0.50 | 401408 | valid |

The first context run is useful diagnostically, but it is not a final comparable
result because 21 predicted crops failed during Qwen inference.

## Corrected Ceiling Comparison

The Kaggle notebook did not have the clean Stage 3 ceiling attached, so the
ceiling was recomputed locally from the clean v7f Stage 3 run.

| Metric | tight 0.15 | context 0.30 capped | context 0.50 uncapped | context 0.50 capped |
| --- | ---: | ---: | ---: | ---: |
| detector match rate | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| good crop rate among matched | 0.9828 | 0.9828 | 0.9828 | 0.9828 |
| VLM correct rate on good pred crops | 0.3684 | 0.4035 | 0.3860 | 0.4035 |
| pipeline correct rate | 0.3621 | 0.3966 | 0.3793 | 0.3966 |
| Stage 3 ceiling correct rate | 0.4655 | 0.4655 | 0.4655 | 0.4655 |
| ceiling minus actual gap | 0.1034 | 0.0690 | 0.0862 | 0.0690 |
| Qwen successful pred-crop outputs | 214/215 | 215/215 | 194/215 | 215/215 |

## Main Error Shift

| Error pattern | tight 0.15 | context 0.30 capped | context 0.50 uncapped | context 0.50 capped |
| --- | ---: | ---: | ---: | ---: |
| `insulator_ok -> defect_flashover` | 15 | 12 | 10 | 15 |
| `insulator_ok -> defect_broken` | 7 | 5 | 5 | 3 |
| `defect_flashover -> insulator_ok` | 3 | 7 | 7 | 8 |
| `defect_flashover -> defect_broken` | 4 | 3 | 5 | 4 |
| `defect_broken -> defect_flashover` | 3 | 4 | 2 | 2 |

Interpretation: the valid capped context runs improve end-to-end accuracy from
21/58 to 23/58 and remove Qwen OOM failures. The `0.30` variant is currently
the better-balanced candidate because it lowers `insulator_ok -> defect_flashover`
relative to tight crops while preserving more flashover recall than `0.50`.

## Next Rerun

The comparable capped reruns are now complete:

- notebook: `notebooks/stage4_context_pad050_kaggle_run.ipynb`
- run name: `stage4_detector_to_vlm_pred_val_context_pad050_maxpix401k_kaggle`
- crop padding: `0.50`
- Qwen visual cap: `max_pixels=401408`
- notebook: `notebooks/stage4_context_pad030_kaggle_run.ipynb`
- run name: `stage4_detector_to_vlm_pred_val_context_pad030_maxpix401k_kaggle`
- crop padding: `0.30`
- Qwen visual cap: `max_pixels=401408`

## Decision

Use context pad030 capped as the current best Stage 4 input strategy candidate.
It is still a trade-off, but it improves total pipeline correctness while being
less damaging to flashover than pad050.

The remaining bottleneck is not crop geometry. It is still VLM coarse-class
separation, especially normal insulators versus flashover-like dark traces.
