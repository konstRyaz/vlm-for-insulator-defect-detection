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
| non_vlm_efficientnet_b0_logreg | efficientnet_b0 | timm | efficientnet_b0.ra_in1k | logreg | 0.1000 |  | balanced | 0.6207 | 0.6062 | 0.6875 | 0.5000 | 0.6667 | ok |
| non_vlm_convnext_tiny_logreg | convnext_tiny | timm | convnext_tiny.fb_in1k | logreg | 1.0000 |  | balanced | 0.6552 | 0.5902 | 0.7188 | 0.6000 | 0.5000 | ok |
| non_vlm_resnet50_logreg | resnet50 | timm | resnet50.a1_in1k | logreg | 1.0000 |  | balanced | 0.4483 | 0.4580 | 0.4375 | 0.4500 | 0.5000 | ok |

## Interpretation placeholder
Compare this table against Qwen direct and DINOv2+Qwen hybrid in `reports/next_research/vlm_benefit/`.