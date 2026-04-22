# Stage 3 Prompt Sweep v6 Checkpoint

- fixed_at: 2026-04-22T20:58:09
- source_archive: `~/Downloads/stage3_deliverables_stage3_qwen_val_v2_sweep_v6.tar.gz`
- extracted_files:
  - `outputs/stage3_prompt_sweep_v6_checkpoint/RESULT_SUMMARY.md`
  - `outputs/stage3_prompt_sweep_v6_checkpoint/aggregate/prompt_sweep_comparison.csv`
  - `outputs/stage3_prompt_sweep_v6_checkpoint/aggregate/prompt_sweep_comparison.md`
- tracked_copies:
  - `reports/stage3_prompt_sweep_v6_checkpoint/RESULT_SUMMARY.md`
  - `reports/stage3_prompt_sweep_v6_checkpoint/prompt_sweep_comparison.csv`
  - `reports/stage3_prompt_sweep_v6_checkpoint/prompt_sweep_comparison.md`

## Best From Previous Sweep
- prompt_version: `qwen_vlm_labels_v1_prompt_v6d_balanced_notaglock`
- verdict: `SOFT_PASS`
- visibility_macro_f1: `0.4935064935064935`
- abs_ambiguous_gap: `0.06896551724137931`
- coarse_acc: `0.9827586206896551`
- tag_mean_jaccard: `0.3395730706075532`
- pred_ambiguous_rate: `0.08620689655172414`
- gt_ambiguous_rate: `0.15517241379310345`

## Top-3 (By Sweep Ranking)
- rank 1: `qwen_vlm_labels_v1_prompt_v6d_balanced_notaglock` | vis_macro_f1=0.4935064935064935 | amb_gap=0.06896551724137931 | coarse_acc=0.9827586206896551 | tag_jaccard=0.3395730706075532
- rank 2: `qwen_vlm_labels_v1_prompt_v6c_balanced` | vis_macro_f1=0.45384615384615384 | amb_gap=0.08620689655172414 | coarse_acc=0.9827586206896551 | tag_jaccard=0.33891625615763543
- rank 3: `qwen_vlm_labels_v1_prompt_v6b_positive_ambiguous` | vis_macro_f1=0.45384615384615384 | amb_gap=0.08620689655172414 | coarse_acc=0.9827586206896551 | tag_jaccard=0.33275862068965506
