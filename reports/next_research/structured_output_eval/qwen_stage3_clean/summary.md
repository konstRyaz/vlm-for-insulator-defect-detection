# Structured output evaluation: qwen_stage3_clean

## Automatic metrics
| metric | value |
|---|---|
| coarse_class_accuracy | 0.4655 |
| visibility_accuracy | 0.8448 |
| needs_review_accuracy | 0.8793 |
| tag_exact_rate | 0.0000 |
| tag_mean_jaccard | 0.1822 |
| description_present_rate | 1.0000 |
| description_generic_auto_rate | 0.0517 |

## Manual rubric summary
| manual_metric | n_scored | mean | score_counts |
|---|---|---|---|
| manual_tag_score | 0 | 0.0000 | {} |
| manual_description_relevance | 0 | 0.0000 | {} |
| manual_visual_evidence_score | 0 | 0.0000 | {} |
| manual_hallucination_score | 0 | 0.0000 | {} |
| manual_usefulness_score | 0 | 0.0000 | {} |
| manual_class_text_consistency | 0 | 0.0000 | {} |

## Artifacts
- details: `reports\next_research\structured_output_eval\qwen_stage3_clean\structured_eval_details.csv`
- metrics: `reports\next_research\structured_output_eval\qwen_stage3_clean\structured_eval_metrics.json`
- manual review template: `reports\next_research\structured_output_eval\qwen_stage3_clean\manual_review_template.csv`

Manual score convention: 0 = bad, 1 = partial/uncertain, 2 = good. For hallucination, 2 means no clear hallucination.