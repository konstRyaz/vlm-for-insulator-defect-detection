# Operation Next Leaderboard

| run_id | stage | type | accuracy | macro_f1 | ok recall | flashover recall | broken recall | note |
|---|---|---|---:|---:|---:|---:|---:|---|
| stage3_qwen_val_v2_clean_final | stage3 | clean_final | 0.4655 | 0.2882_eval5 / 0.4804_3class | 0.3438 | 0.6500 | 0.5000 | clean Qwen reporter baseline; 3-class macro from paired script differs from evaluator macro including unknown/other |
| stage4_context_pad030_maxpix401k | stage4 | clean_final | 0.3966_pipeline_correct |  |  |  |  | best Stage4 actual context crop baseline |
| stage3_clip_train_selected_clean | stage3_coarse_only | clean_final | 0.5345 | 0.3713 | 0.5938 | 0.6000 | 0.0000 | coarse-only classifier; strongest clean discriminative signal but broken recall zero |
| stage3_hybrid_hard_clip_qwen | stage3_hybrid | clean_final_partial_system | 0.5345 | 0.2228_eval5 / 0.3713_3class | 0.5938 | 0.6000 | 0.0000 | improves accuracy but fails full-system gate because broken recall is zero and macro-F1 drops vs Qwen 3-class |
| stage3_qwen25vl_3b_lora_masked_smoke_clean | stage3_lora | failed_preflight | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | overfit gate failed; repeated punctuation; full val skipped |
| stage3_qwen25vl_3b_lora_overfit_gate_clean_v25 | stage3_lora | smoke_failed_semantic_gate | 0.4000_overfit | not_comparable_overfit_smoke | 0.0000_overfit | 1.0000_overfit | 0.0000_overfit | parse gate fixed: 5/5 valid JSON; semantic gate failed: 2/5 coarse correct, all predictions collapsed to defect_flashover; full val skipped |
