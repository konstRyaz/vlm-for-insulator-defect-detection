# Stage 4 DINOv2 + Qwen Light Hybrid Checkpoint

This checkpoint evaluates a lightweight hybrid on the clean Stage 4 detector-to-VLM path. The run reuses frozen detector predictions and frozen Qwen reporter outputs, then replaces only `coarse_class` with a DINOv2 linear-probe classifier trained on clean train crops.

## Main Result

- Previous Stage 4 Qwen reporter baseline: `23/58 = 0.3966` pipeline-correct.
- DINOv2 hard hybrid: `28/58 = 0.4828` pipeline-correct.
- Delta: `+5` GT objects, `+0.0862` absolute.

This passes the strong Stage 4 gate from the operation plan (`>=27/58`).

## Class Behavior

- `insulator_ok`: `6/32 = 0.1875` recall.
- `defect_flashover`: `19/20 = 0.9500` recall.
- `defect_broken`: `3/6 = 0.5000` recall.

Interpretation: the hybrid strongly repairs flashover recall, but it overpredicts flashover on normal insulators. The next useful work is not another VLM prompt sweep, but calibration of the discriminative coarse branch.

## Diagnostic Policy Probe

The table below is exploratory and uses the current val slice, so it is not a final clean selection. It is useful for choosing the next pre-registered experiment.

| policy | correct | acc | macro3 | recall_insulator_ok | recall_defect_flashover | recall_defect_broken | pred_flash | pred_ok | pred_broken |
|---|---|---|---|---|---|---|---|---|---|
| qwen_ok_veto_flash_lt_0.34 | 29 | 0.5000 | 0.4716 | 0.2188 | 0.9500 | 0.5000 | 43 | 8 | 7 |
| flash_conf_ge_0.34_else_qwen | 29 | 0.5000 | 0.4638 | 0.2188 | 0.9500 | 0.5000 | 42 | 8 | 8 |
| qwen_ok_veto_flash_lt_0.38 | 28 | 0.4828 | 0.4774 | 0.4062 | 0.6000 | 0.5000 | 30 | 21 | 7 |
| qwen_ok_veto_flash_lt_0.40 | 28 | 0.4828 | 0.4774 | 0.4062 | 0.6000 | 0.5000 | 30 | 21 | 7 |
| qwen_ok_veto_flash_lt_0.45 | 28 | 0.4828 | 0.4774 | 0.4062 | 0.6000 | 0.5000 | 30 | 21 | 7 |
| qwen_ok_veto_flash_lt_0.50 | 28 | 0.4828 | 0.4774 | 0.4062 | 0.6000 | 0.5000 | 30 | 21 | 7 |
| dinov2_hard | 28 | 0.4828 | 0.4543 | 0.1875 | 0.9500 | 0.5000 | 44 | 7 | 7 |
| any_conf_ge_0.30 | 28 | 0.4828 | 0.4543 | 0.1875 | 0.9500 | 0.5000 | 44 | 7 | 7 |
| any_conf_ge_0.32 | 28 | 0.4828 | 0.4543 | 0.1875 | 0.9500 | 0.5000 | 44 | 7 | 7 |
| any_conf_ge_0.33 | 28 | 0.4828 | 0.4543 | 0.1875 | 0.9500 | 0.5000 | 44 | 7 | 7 |
| flash_conf_ge_0.33_else_qwen | 28 | 0.4828 | 0.4543 | 0.1875 | 0.9500 | 0.5000 | 44 | 7 | 7 |
| any_conf_ge_0.34 | 28 | 0.4828 | 0.4467 | 0.1562 | 1.0000 | 0.5000 | 45 | 6 | 7 |

## Recommended Next Experiment

Freeze the hard DINOv2 hybrid as a strong candidate, then run one calibrated-policy notebook where the flashover-vs-ok veto rule is selected by train/CV only and applied once to val. The main candidate rule is: keep the DINOv2 prediction, except when Qwen predicts `insulator_ok` and DINOv2 predicts `defect_flashover` with low confidence.
