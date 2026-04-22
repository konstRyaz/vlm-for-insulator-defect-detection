# Stage 3 v6d vs v6f Checkpoint

- fixed_at: 2026-04-22T22:05:17
- source_archive: `~/Downloads/stage3_deliverables_stage3_qwen_val_v2_v6d_vs_v6f.tar.gz`
- tracked_files:
  - `reports/stage3_v6d_vs_v6f_checkpoint/RESULT_SUMMARY.md`
  - `reports/stage3_v6d_vs_v6f_checkpoint/selection_decision.json`
  - `reports/stage3_v6d_vs_v6f_checkpoint/v6d_vs_v6f_comparison.csv`
  - `reports/stage3_v6d_vs_v6f_checkpoint/v6d_vs_v6f_comparison.md`

## Decision
- selected_prompt: `qwen_vlm_labels_v1_prompt_v6d_balanced_notaglock`
- recommendation: `freeze_v6d_stop_prompt_tuning`
- rule_visibility_macro_f1_ge_v6d: `False`
- rule_abs_ambiguous_gap_lt_v6d: `False`
- rule_coarse_acc_no_regress: `True`
- rule_pred_ambiguous_rate_le_0_15: `True`

## Metrics
| prompt_version | coarse_acc | coarse_macro_f1 | visibility_acc | visibility_macro_f1 | needs_review_acc | tag_mean_jaccard | pred_ambiguous_rate | gt_ambiguous_rate | abs_ambiguous_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen_vlm_labels_v1_prompt_v6d_balanced_notaglock | 0.982759 | 0.581441 | 0.844828 | 0.493506 | 0.896552 | 0.339573 | 0.086207 | 0.155172 | 0.068966 |
| qwen_vlm_labels_v1_prompt_v6f_balanced_notaglock_soft_ambiguous_raise | 0.982759 | 0.581441 | 0.827586 | 0.453846 | 0.879310 | 0.338916 | 0.068966 | 0.155172 | 0.086207 |
