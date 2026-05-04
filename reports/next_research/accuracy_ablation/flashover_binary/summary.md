# Non-VLM baseline sweep

This report compares frozen visual feature extractors plus classical classifiers against VLM/hybrid systems.
Hyperparameters were selected by train CV only. Validation was evaluated once per selected configuration.

## Class distribution
| split | label | count |
|---|---|---|
| train | insulator_ok | 80 |
| train | defect_flashover | 10 |
| val | insulator_ok | 32 |
| val | defect_flashover | 20 |

## Leaderboard
| run_id | model_key | kind | model_id | classifier | C | kernel | class_weight | accuracy | macro_f1 | recall_insulator_ok | recall_defect_flashover | recall_defect_broken | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| non_vlm_dinov2_binary_logreg | dinov2_binary | hf_auto | facebook/dinov2-base | logreg | 0.0300 |  | balanced | 0.7115 | 0.7026 | 0.7188 | 0.7000 | 0.0000 | ok |
| non_vlm_clip_b32_binary_logreg | clip_b32_binary | hf_auto | openai/clip-vit-base-patch32 | logreg | 0.0100 |  | balanced | 0.6154 | 0.6061 | 0.6250 | 0.6000 | 0.0000 | ok |

## Interpretation placeholder
Compare this table against Qwen direct and DINOv2+Qwen hybrid in `reports/next_research/vlm_benefit/`.