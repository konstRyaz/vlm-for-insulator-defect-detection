# Stage 4 Clean v7f Checkpoint

This checkpoint records the leakage-free Stage 4 result after the final clean
Stage 3 prompt selection.

## Validity

- Stage 3 / Stage 4 prompt version: `qwen_vlm_labels_v1_prompt_v7f_flashover_unclear_to_unknown_nocroppath`
- VLM model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Stage 4 VLM run summary confirms `user_prompt_contains_crop_path_token = false`
- Stage 4 VLM attempted `215` predicted crops, with `214` successful outputs and `1` backend error
- Historical runs with prompt-visible `crop_path` remain diagnostic only

## Main Metrics

| Metric | Value |
| --- | ---: |
| GT objects | 58 |
| Detector match rate | 1.0000 |
| Good crop rate among matched | 0.9828 |
| VLM correct rate on good predicted crops | 0.3684 |
| Stage 4 actual pipeline correct rate | 0.3621 |
| Stage 3 clean ceiling correct rate | 0.4655 |
| Ceiling minus actual gap | 0.1034 |

## Error Buckets

| Bucket | Count | Rate |
| --- | ---: | ---: |
| detector_miss | 0 | 0.0000 |
| bad_crop_from_detector | 1 | 0.0172 |
| vlm_error_on_good_pred_crop | 36 | 0.6207 |
| routing_or_filtering_error | 0 | 0.0000 |
| correct_pipeline_hit | 21 | 0.3621 |

## Main Reading

The detector side is not the bottleneck on this validation run: every GT object is matched and only one matched crop falls below the good-crop IoU threshold.

Most remaining loss is in VLM coarse-class decisions on geometrically good predicted crops. The largest failure pattern is `insulator_ok -> defect_flashover` with `15` cases.

## Key Artifacts

- Clean Stage 3 final artifacts: `outputs/_external_runs/stage3_clean_final_v7f/`
- Clean Stage 4 eval with ceiling: `outputs/_external_runs/stage4_clean_v7f_with_ceiling/04_eval/`
- Clean Stage 4 visual report: `outputs/_external_runs/stage4_clean_v7f_with_ceiling/04_eval/visuals/report.md`
- Targeted error review: `outputs/_external_runs/stage4_clean_v7f_with_ceiling/04_eval/targeted_review/targeted_error_review.html`
- Main case table: `outputs/_external_runs/stage4_clean_v7f_with_ceiling/04_eval/stage4_case_table.csv`

## Research Use

Use this checkpoint as the current clean Stage 4 reference for final tables and discussion.

The clean comparison is now:

`GT crop -> VLM ceiling = 0.4655`

`pred crop -> VLM actual = 0.3621`

`gap = 0.1034`
