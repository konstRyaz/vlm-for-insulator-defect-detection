# Non-VLM baseline sweep

This report compares frozen visual feature extractors plus classical classifiers against VLM/hybrid systems.
Hyperparameters were selected by train CV only. Validation was evaluated once per selected configuration.

## Class distribution
| split | label | count |
|---|---|---|
| train | insulator_ok | 15 |
| train | defect_flashover | 3 |
| train | defect_broken | 2 |
| val | insulator_ok | 6 |
| val | defect_flashover | 0 |
| val | defect_broken | 4 |

## Leaderboard
_No rows._

## Interpretation placeholder
Compare this table against Qwen direct and DINOv2+Qwen hybrid in `reports/next_research/vlm_benefit/`.