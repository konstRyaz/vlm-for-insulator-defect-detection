# Stage 3 DINOv2 Train-CV Policy Clean

| policy                           |   selected_threshold |   accuracy |   correct |   total |   macro_f1 |   recall_insulator_ok |   recall_defect_flashover |   recall_defect_broken |
|:---------------------------------|---------------------:|-----------:|----------:|--------:|-----------:|----------------------:|--------------------------:|-----------------------:|
| flash_lowconf_second_best_cv0p35 |                 0.35 |   0.551724 |        32 |      58 |   0.576554 |               0.40625 |                      0.65 |               1        |
| hard_dinov2                      |                 0.35 |   0.5      |        29 |      58 |   0.505263 |               0.1875  |                      0.95 |               0.666667 |

## Train OOF Selection

| policy                    | threshold   |   accuracy |   macro_f1 |
|:--------------------------|:------------|-----------:|-----------:|
| flash_lowconf_second_best | 0.35        |   0.638095 |   0.562477 |
| flash_lowconf_second_best | 0.34        |   0.609524 |   0.523443 |
| hard_train_oof            |             |   0.590476 |   0.516962 |
| flash_lowconf_second_best | 0.3         |   0.590476 |   0.516962 |
| flash_lowconf_second_best | 0.32        |   0.590476 |   0.516962 |
| flash_lowconf_second_best | 0.33        |   0.590476 |   0.516962 |
| flash_lowconf_second_best | 0.36        |   0.609524 |   0.421195 |
| flash_lowconf_second_best | 0.38        |   0.628571 |   0.380457 |
| flash_lowconf_second_best | 0.4         |   0.628571 |   0.380457 |
| flash_lowconf_second_best | 0.45        |   0.628571 |   0.380457 |
| flash_lowconf_second_best | 0.5         |   0.628571 |   0.380457 |
