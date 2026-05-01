# Stage 4 Detector -> VLM Summary

- GT objects: 58
- Detector match rate: 1.0000
- Good crop rate among matched: 0.9828
- VLM correct rate among good pred crops: 0.5965
- Pipeline correct rate: 0.5862
- Ceiling correct rate: 0.0000
- Ceiling vs actual gap: -0.5862

## Error buckets
- detector_miss: 0 (0.0000)
- bad_crop_from_detector: 1 (0.0172)
- vlm_error_on_good_pred_crop: 23 (0.3966)
- routing_or_filtering_error: 0 (0.0000)
- correct_pipeline_hit: 34 (0.5862)

## Artifacts
- stage4_metrics.json: `/kaggle/working/vlm-for-insulator-defect-detection/outputs/stage4/stage4_hybrid_dinov2_qwen_traincv_policy_pad030/04_eval_hybrid/dinov2_flash_lowconf_second_best_cv0p35/stage4_metrics.json`
- stage4_error_breakdown.json: `/kaggle/working/vlm-for-insulator-defect-detection/outputs/stage4/stage4_hybrid_dinov2_qwen_traincv_policy_pad030/04_eval_hybrid/dinov2_flash_lowconf_second_best_cv0p35/stage4_error_breakdown.json`
- stage4_case_table.csv: `/kaggle/working/vlm-for-insulator-defect-detection/outputs/stage4/stage4_hybrid_dinov2_qwen_traincv_policy_pad030/04_eval_hybrid/dinov2_flash_lowconf_second_best_cv0p35/stage4_case_table.csv`
- ceiling_vs_actual.json: `/kaggle/working/vlm-for-insulator-defect-detection/outputs/stage4/stage4_hybrid_dinov2_qwen_traincv_policy_pad030/04_eval_hybrid/dinov2_flash_lowconf_second_best_cv0p35/ceiling_vs_actual.json`