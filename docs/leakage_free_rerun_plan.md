# Leakage-Free Rerun Plan

This note defines which Stage 3 / Stage 4 experiments must be rerun after the `crop_path` leakage finding.

## Why reruns are required

Historical Stage 3 and Stage 4 prompt paths exposed `crop_path` inside the user prompt.

That is unsafe because:

- Stage 3 GT crop paths encode the GT coarse class via folder names such as `crops/val/defect_broken/...`
- Stage 4 predicted crop paths encode the detector class via folder names such as `crops/val/insulator_ok/...`

So any run that used prompt-visible `crop_path` is preserved only as a diagnostic checkpoint, not as a final research result.

## Which historical results are deprecated for final reporting

- Stage 3 GT-crop ceiling runs that used prompt versions without `_nocroppath`
- Stage 3 prompt sweep / micro-ablation checkpoints that used prompt versions without `_nocroppath`
- Stage 4 detector-to-VLM runs that used prompt-visible `crop_path`

These runs can still be cited as debugging history, but final tables and conclusions should come only from the clean reruns below.

## Clean prompt policy

Use only prompt versions with the `_nocroppath` suffix.

Current safe default:

- `qwen_vlm_labels_v1_prompt_v6d_balanced_notaglock_nocroppath`

The Stage 3 runner now records this audit field in `run_summary.json`:

- `prompt_selection.user_prompt_contains_crop_path_token`

For a valid clean run this field must be `false`.

## Clean rerun notebook order

### 1) Clean Stage 3 prompt sweep

Notebook:

- `notebooks/stage3_prompt_sweep_visibility_v6_clean.ipynb`

Purpose:

- rerun the visibility-focused prompt family without `crop_path`
- produce a clean comparison table for v3/v4/v5a/v6a/v6b/v6c/v6d/v6e

Expected archive:

- `/kaggle/working/stage3_deliverables_stage3_qwen_val_v2_sweep_v6_clean.tar.gz`

### 2) Clean Stage 3 final micro-ablation

Notebook:

- `notebooks/stage3_final_micro_ablation_v6d_vs_v6f_clean.ipynb`

Purpose:

- rerun the final `v6d` vs `v6f` comparison without `crop_path`

Expected archive:

- `/kaggle/working/stage3_deliverables_stage3_qwen_val_v2_v6d_vs_v6f_clean.tar.gz`

### 3) Clean Stage 3 final GT-crop baseline

Notebook:

- `notebooks/stage3_qwen_kaggle_clean_onepass.ipynb`

Purpose:

- produce the clean GT-crop ceiling run, eval package, and visuals
- provide the clean Stage 3 ceiling for Stage 4 comparison
- after the clean sweep/micro-ablation, update only the notebook `PROMPT_VERSION` field if the winning prompt differs from the default

Stable run id:

- `stage3_qwen_val_v2_clean_final`

Expected archive:

- `/kaggle/working/stage3_deliverables_stage3_qwen_val_v2_clean_final.tar.gz`

### 4) Clean Stage 4 detector -> VLM run

Notebook:

- `notebooks/stage4_detector_to_vlm_kaggle_run.ipynb`

Purpose:

- run `pred bbox -> VLM` with the clean no-`crop_path` prompt
- if the clean Stage 3 ceiling run is attached as a Kaggle input, the notebook can also reuse it for `ceiling vs actual`
- before this run, set the notebook/config prompt version to the winner of the clean Stage 3 rerun path

Expected archive:

- `/kaggle/working/stage4_deliverables_stage4_detector_to_vlm_pred_val_kaggle.tar.gz`

## Minimal acceptance checks after every rerun

For Stage 3 clean runs:

- selected prompt version ends with `_nocroppath`
- `run_summary.json -> prompt_selection.user_prompt_contains_crop_path_token == false`
- schema validation still passes

For Stage 4 clean runs:

- selected prompt version is `qwen_vlm_labels_v1_prompt_v6d_balanced_notaglock_nocroppath`
- notebook preflight confirms the selected user prompt does not contain `crop_path`
- if a clean ceiling run is attached, `ceiling_vs_actual.json` is based on `stage3_qwen_val_v2_clean_final`

## Final reporting rule

Use only the clean rerun artifacts above for:

- Stage 3 final metrics
- prompt-selection conclusions
- Stage 4 `ceiling vs actual`
- thesis tables and plots

Historical leakage-affected checkpoints stay in the repository only as troubleshooting history.
