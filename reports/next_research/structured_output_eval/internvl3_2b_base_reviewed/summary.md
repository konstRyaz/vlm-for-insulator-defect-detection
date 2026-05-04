# Structured output evaluation: internvl3_2b_base_reviewed

## Automatic metrics
| metric | value |
|---|---|
| coarse_class_accuracy | 0.5517 |
| visibility_accuracy | 0.7931 |
| needs_review_accuracy | 0.8448 |
| tag_exact_rate | 0.0172 |
| tag_mean_jaccard | 0.0330 |
| description_present_rate | 1.0000 |
| description_generic_auto_rate | 0.2586 |

## Manual rubric summary
| manual_metric | n_scored | mean | score_counts |
|---|---|---|---|
| manual_tag_score | 58 | 0.0862 | {'0': 54, '1': 3, '2': 1} |
| manual_description_relevance | 58 | 1.4828 | {'1': 30, '2': 28} |
| manual_visual_evidence_score | 58 | 0.0862 | {'0': 54, '1': 3, '2': 1} |
| manual_hallucination_score | 58 | 0.5345 | {'1': 15, '0': 35, '2': 8} |
| manual_usefulness_score | 58 | 1.4828 | {'1': 30, '2': 28} |
| manual_class_text_consistency | 58 | 0.7931 | {'2': 23, '0': 35} |

## Artifacts
- details: `reports\next_research\structured_output_eval\internvl3_2b_base_reviewed\structured_eval_details.csv`
- metrics: `reports\next_research\structured_output_eval\internvl3_2b_base_reviewed\structured_eval_metrics.json`
- manual review template: `reports\next_research\structured_output_eval\internvl3_2b_base_reviewed\manual_review_template.csv`

Manual score convention: 0 = bad, 1 = partial/uncertain, 2 = good. For hallucination, 2 means no clear hallucination.