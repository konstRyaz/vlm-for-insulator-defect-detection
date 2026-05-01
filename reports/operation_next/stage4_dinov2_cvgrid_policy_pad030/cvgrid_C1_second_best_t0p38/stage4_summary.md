# Stage 4 Detector -> VLM Summary

- GT objects: 58
- Detector match rate: 1.0000
- Good crop rate among matched: 0.9828
- VLM correct rate among good pred crops: 0.5439
- Pipeline correct rate: 0.5345
- Ceiling correct rate: 0.0000
- Ceiling vs actual gap: -0.5345

## Error buckets
- detector_miss: 0 (0.0000)
- bad_crop_from_detector: 1 (0.0172)
- vlm_error_on_good_pred_crop: 26 (0.4483)
- routing_or_filtering_error: 0 (0.0000)
- correct_pipeline_hit: 31 (0.5345)

## Artifacts
- stage4_metrics.json: `/kaggle/working/vlm-for-insulator-defect-detection/outputs/stage4/stage4_hybrid_dinov2_qwen_cvgrid_policy_pad030/04_eval_hybrid/cvgrid_C1_second_best_t0p38/stage4_metrics.json`
- stage4_error_breakdown.json: `/kaggle/working/vlm-for-insulator-defect-detection/outputs/stage4/stage4_hybrid_dinov2_qwen_cvgrid_policy_pad030/04_eval_hybrid/cvgrid_C1_second_best_t0p38/stage4_error_breakdown.json`
- stage4_case_table.csv: `/kaggle/working/vlm-for-insulator-defect-detection/outputs/stage4/stage4_hybrid_dinov2_qwen_cvgrid_policy_pad030/04_eval_hybrid/cvgrid_C1_second_best_t0p38/stage4_case_table.csv`
- ceiling_vs_actual.json: `/kaggle/working/vlm-for-insulator-defect-detection/outputs/stage4/stage4_hybrid_dinov2_qwen_cvgrid_policy_pad030/04_eval_hybrid/cvgrid_C1_second_best_t0p38/ceiling_vs_actual.json`