# VLM vs non-VLM benefit report

## Main question

Does the VLM module improve the insulator-defect pipeline compared with ordinary no-VLM computer-vision baselines?

The answer must be decomposed: class accuracy and structured-reporting value are different axes.

## Classification comparison

| system | uses_vlm | uses_train | output_type | accuracy | macro_f1 | recommended_role |
|---|---|---|---|---|---|---|
| best_non_vlm_classifier | no | yes_classifier_train_cv | class_only | 0.6552 | 0.6684 | class-only CV baseline |
| stage3_qwen_val_v2_clean_final | yes | no_zero_shot | JSON+class | 0.4655 | 0.2882_eval5 / 0.4804_3class | direct/frozen VLM structured reporter |
| stage4_context_pad030_maxpix401k | yes | no_zero_shot | JSON+class | 0.3966 |  | direct/frozen VLM structured reporter |
| stage4_dinov2_packfix_secondbest035 | partial_qwen_reporter | yes_classifier_branch | JSON+class | 0.5862 | 0.5922_3class / 0.4441_all_with_nan | hybrid classifier+reporter |
| stage3_vlm_backbone_internvl3_2b_base | yes | no_zero_shot | JSON+class | 0.5517 | 0.2853 | direct/frozen VLM structured reporter |

## Structured-output comparison

| run_name | coarse_class_accuracy | visibility_accuracy | tag_mean_jaccard | description_present_rate | description_generic_auto_rate | manual_description_relevance_mean | manual_hallucination_score_mean | manual_usefulness_score_mean |
|---|---|---|---|---|---|---|---|---|
| hybrid_dinov2_qwen_champion | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hybrid_dinov2_qwen_champion_matched | 0.5862 | 0.7931 | 0.1443 | 1.0000 | 0.0172 | 0.0000 | 0.0000 | 0.0000 |
| hybrid_dinov2_qwen_champion_matched_reviewed | 0.5862 | 0.7931 | 0.1443 | 1.0000 | 0.0172 | 1.5862 | 0.9655 | 1.5862 |
| internvl3_2b_base | 0.5517 | 0.7931 | 0.0330 | 1.0000 | 0.2586 | 0.0000 | 0.0000 | 0.0000 |
| internvl3_2b_base_reviewed | 0.5517 | 0.7931 | 0.0330 | 1.0000 | 0.2586 | 1.4828 | 0.5345 | 1.4828 |
| qwen_stage3_clean | 0.4655 | 0.8448 | 0.1822 | 1.0000 | 0.0517 | 0.0000 | 0.0000 | 0.0000 |
| qwen_stage3_clean_reviewed | 0.4655 | 0.8448 | 0.1822 | 1.0000 | 0.0517 | 1.4483 | 0.9828 | 1.4483 |
| qwen_stage4_context_pad030 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| qwen_stage4_context_pad030_matched | 0.3966 | 0.7931 | 0.1443 | 1.0000 | 0.0172 | 0.0000 | 0.0000 | 0.0000 |
| qwen_stage4_context_pad030_matched_reviewed | 0.3966 | 0.7931 | 0.1443 | 1.0000 | 0.0172 | 1.3966 | 0.8966 | 1.3966 |

## Interpretation template

1. If no-VLM classifier is best by class accuracy, report it honestly.
2. The VLM benefit is not necessarily raw accuracy; it is JSON/reporting/explanation output.
3. The most defensible architecture is hybrid: visual classifier for `coarse_class`, VLM reporter for structured fields.
4. If Qwen text/tags disagree with the DINOv2-overridden class, add a consistency/review flag or regenerate reporter fields conditioned on class.

## Recommended claim

> Direct frozen VLM is not the strongest crop-level classifier on this small specialized dataset. However, VLM remains useful as a structured reporter. The hybrid system combines the class accuracy of discriminative visual features with the reportability of VLM output.