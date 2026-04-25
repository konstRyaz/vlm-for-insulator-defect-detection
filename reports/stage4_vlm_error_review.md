# Stage 4 Error Review (clean v7f)

This note summarizes the current leakage-free Stage 4 result on val after
detector crops are passed into the final clean Stage 3 VLM prompt.

Earlier Stage 4 reviews that used prompt-visible `crop_path` are diagnostic
history only. They should not be used for final metrics.

## Main Numbers

- GT objects: 58
- Detector match rate: 1.0000
- Good crop rate among matched: 0.9828
- VLM correct rate among good pred crops: 0.3684
- Pipeline correct rate (actual): 0.3621
- Ceiling correct rate (clean Stage 3 GT crop): 0.4655
- Ceiling - actual gap: 0.1034

## Where The Drop Comes From

- `detector_miss`: 0
- `bad_crop_from_detector`: 1
- `vlm_error_on_good_pred_crop`: 36
- `routing_or_filtering_error`: 0
- `correct_pipeline_hit`: 21

Detector localization is strong in this run. The remaining loss is mostly from
VLM coarse-class decisions on good predicted crops.

## Coarse Error Pattern

On good predicted crops, the largest failure pattern is:

`insulator_ok -> defect_flashover`: 15 cases

Other notable errors:

- `insulator_ok -> defect_broken`: 7
- `defect_flashover -> defect_broken`: 4
- `defect_broken -> defect_flashover`: 3
- `defect_flashover -> insulator_ok`: 3
- `defect_flashover -> unknown`: 2

## Leakage Check

The Stage 4 pred-crop VLM run used:

`qwen_vlm_labels_v1_prompt_v7f_flashover_unclear_to_unknown_nocroppath`

The run summary confirms:

- `user_prompt_contains_crop_path_token = false`
- `records_attempted = 215`
- `status_ok = 214`
- `status_backend_error = 1`

The earlier detector-label leakage signal is no longer present.

## Current Interpretation

The clean Stage 4 result is lower than the clean Stage 3 ceiling, but the gap is
now moderate rather than catastrophic:

`0.4655 - 0.3621 = 0.1034`

This supports the current research story: the detector-to-crop handoff is
geometrically reliable on this split, while the main challenge is stable
coarse-class visual interpretation by the VLM.
