# Stage 3 Prompt Sweep Comparison

- control_version: `qwen_vlm_labels_v1_prompt_v5a_visibility_gate_best`
- best_prompt_version: `qwen_vlm_labels_v1_prompt_v6d_balanced_notaglock`
- best_run_id: `stage3_qwen_val_v2_sweep_v6_qwen_vlm_labels_v1_prompt_v6d_balanced_notaglock`
- best_verdict: `SOFT_PASS`

| prompt_version | run_id | parse_success | schema_valid | coarse_acc | coarse_macro_f1 | visibility_acc | visibility_macro_f1 | needs_review_acc | tag_exact | tag_mean_jaccard | pred_ambiguous_rate | gt_ambiguous_rate | backend | model | abs_ambiguous_gap | verdict | rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen_vlm_labels_v1_prompt_v6d_balanced_notaglock | stage3_qwen_val_v2_sweep_v6_qwen_vlm_labels_v1_prompt_v6d_balanced_notaglock | 1.000000 | 1.000000 | 0.982759 | 0.581441 | 0.844828 | 0.493506 | 0.896552 | 0.051724 | 0.339573 | 0.086207 | 0.155172 | qwen_hf | Qwen/Qwen2.5-VL-3B-Instruct | 0.068966 | SOFT_PASS | 1 |
| qwen_vlm_labels_v1_prompt_v6c_balanced | stage3_qwen_val_v2_sweep_v6_qwen_vlm_labels_v1_prompt_v6c_balanced | 1.000000 | 1.000000 | 0.982759 | 0.581441 | 0.827586 | 0.453846 | 0.879310 | 0.051724 | 0.338916 | 0.068966 | 0.155172 | qwen_hf | Qwen/Qwen2.5-VL-3B-Instruct | 0.086207 | SOFT_PASS | 2 |
| qwen_vlm_labels_v1_prompt_v6b_positive_ambiguous | stage3_qwen_val_v2_sweep_v6_qwen_vlm_labels_v1_prompt_v6b_positive_ambiguous | 1.000000 | 1.000000 | 0.982759 | 0.591947 | 0.827586 | 0.453846 | 0.879310 | 0.034483 | 0.332759 | 0.068966 | 0.155172 | qwen_hf | Qwen/Qwen2.5-VL-3B-Instruct | 0.086207 | SOFT_PASS | 3 |
| qwen_vlm_labels_v1_prompt_v6e_partial_geometry | stage3_qwen_val_v2_sweep_v6_qwen_vlm_labels_v1_prompt_v6e_partial_geometry | 1.000000 | 1.000000 | 0.982759 | 0.591947 | 0.827586 | 0.421866 | 0.879310 | 0.051724 | 0.338259 | 0.034483 | 0.155172 | qwen_hf | Qwen/Qwen2.5-VL-3B-Instruct | 0.120690 | FAIL | 4 |
| qwen_vlm_labels_v1_prompt_v5a_visibility_gate_best | stage3_qwen_val_v2_sweep_v6_qwen_vlm_labels_v1_prompt_v5a_visibility_gate_best | 1.000000 | 1.000000 | 0.982759 | 0.581441 | 0.827586 | 0.421866 | 0.879310 | 0.034483 | 0.334729 | 0.034483 | 0.155172 | qwen_hf | Qwen/Qwen2.5-VL-3B-Instruct | 0.120690 | FAIL | 5 |
| qwen_vlm_labels_v1_prompt_v6a_notaglock | stage3_qwen_val_v2_sweep_v6_qwen_vlm_labels_v1_prompt_v6a_notaglock | 1.000000 | 1.000000 | 0.982759 | 0.591947 | 0.827586 | 0.421866 | 0.879310 | 0.034483 | 0.333087 | 0.034483 | 0.155172 | qwen_hf | Qwen/Qwen2.5-VL-3B-Instruct | 0.120690 | FAIL | 6 |
| qwen_vlm_labels_v1_prompt_v3 | stage3_qwen_val_v2_sweep_v6_qwen_vlm_labels_v1_prompt_v3 | 1.000000 | 1.000000 | 0.982759 | 0.581441 | 0.500000 | 0.385281 | 0.879310 | 0.172414 | 0.374302 | 0.034483 | 0.155172 | qwen_hf | Qwen/Qwen2.5-VL-3B-Instruct | 0.120690 | FAIL | 7 |
| qwen_vlm_labels_v1_prompt_v4 | stage3_qwen_val_v2_sweep_v6_qwen_vlm_labels_v1_prompt_v4 | 1.000000 | 1.000000 | 0.982759 | 0.581441 | 0.551724 | 0.316667 | 0.586207 | 0.137931 | 0.363670 | 0.396552 | 0.155172 | qwen_hf | Qwen/Qwen2.5-VL-3B-Instruct | 0.241379 | FAIL | 8 |