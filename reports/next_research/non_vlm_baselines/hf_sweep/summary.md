# Non-VLM baseline sweep

This report compares frozen visual feature extractors plus classical classifiers against VLM/hybrid systems.
Hyperparameters were selected by train CV only. Validation was evaluated once per selected configuration.

## Class distribution
| split | label | count |
|---|---|---|
| train | insulator_ok | 80 |
| train | defect_flashover | 10 |
| train | defect_broken | 15 |
| val | insulator_ok | 32 |
| val | defect_flashover | 20 |
| val | defect_broken | 6 |

## Leaderboard
| run_id | model_key | kind | model_id | classifier | C | kernel | class_weight | accuracy | macro_f1 | recall_insulator_ok | recall_defect_flashover | recall_defect_broken | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| non_vlm_dinov2_base_logreg | dinov2_base | hf_auto | facebook/dinov2-base | logreg | 0.0300 |  | balanced | 0.6552 | 0.6684 | 0.6562 | 0.6500 | 0.6667 | ok |
| non_vlm_siglip_b16_224_logreg | siglip_b16_224 | hf_auto | google/siglip-base-patch16-224 | logreg | 0.0300 |  | balanced | 0.4828 | 0.5300 | 0.3750 | 0.6000 | 0.6667 | ok |
| non_vlm_clip_l14_logreg | clip_l14 | hf_auto | openai/clip-vit-large-patch14 | logreg | 3.0000 |  | none | 0.5172 | 0.4739 | 0.6562 | 0.3000 | 0.5000 | ok |
| non_vlm_clip_b32_logreg | clip_b32 | hf_auto | openai/clip-vit-base-patch32 | logreg | 0.0300 |  | balanced | 0.5690 | 0.4609 | 0.7500 | 0.4000 | 0.1667 | ok |

## Interpretation placeholder
Compare this table against Qwen direct and DINOv2+Qwen hybrid in `reports/next_research/vlm_benefit/`.