# Structured output evaluation: qwen_stage4_context_pad030_matched_reviewed

## Automatic metrics
| metric | value |
|---|---|
| coarse_class_accuracy | 0.3966 |
| visibility_accuracy | 0.7931 |
| needs_review_accuracy | 0.8448 |
| tag_exact_rate | 0.0000 |
| tag_mean_jaccard | 0.1443 |
| description_present_rate | 1.0000 |
| description_generic_auto_rate | 0.0172 |

## Manual rubric summary
| manual_metric | n_scored | mean | score_counts |
|---|---|---|---|
| manual_tag_score | 58 | 0.4828 | {'0': 35, '1': 18, '2': 5} |
| manual_description_relevance | 58 | 1.3966 | {'1': 35, '2': 23} |
| manual_visual_evidence_score | 58 | 0.4828 | {'0': 35, '1': 18, '2': 5} |
| manual_hallucination_score | 58 | 0.8966 | {'1': 28, '0': 18, '2': 12} |
| manual_usefulness_score | 58 | 1.3966 | {'1': 35, '2': 23} |
| manual_class_text_consistency | 58 | 1.3793 | {'2': 40, '0': 18} |

## Artifacts
- details: `reports\next_research\structured_output_eval\qwen_stage4_context_pad030_matched_reviewed\structured_eval_details.csv`
- metrics: `reports\next_research\structured_output_eval\qwen_stage4_context_pad030_matched_reviewed\structured_eval_metrics.json`
- manual review template: `reports\next_research\structured_output_eval\qwen_stage4_context_pad030_matched_reviewed\manual_review_template.csv`

Manual score convention: 0 = bad, 1 = partial/uncertain, 2 = good. For hallucination, 2 means no clear hallucination.