# Stage 3 CLIP Hybrid Sweep Clean

CLIP model: `openai/clip-vit-base-patch32`

This is an exploratory coarse-only sweep over simple classifiers and thresholds. It uses val labels for comparison, so it should guide hypotheses rather than define a final tuned method.

## Top 12 by macro-F1

| method                                      |   accuracy |   macro_f1_3class |   ok_recall |   flashover_recall |   broken_recall |      C | class_weight   |   flash_thr |
|:--------------------------------------------|-----------:|------------------:|------------:|-------------------:|----------------:|-------:|:---------------|------------:|
| logreg_c0.03_balanced                       |   0.482759 |          0.479894 |     0.28125 |               0.85 |        0.333333 |   0.03 | balanced       |      nan    |
| logreg_c0.3_balanced                        |   0.465517 |          0.472213 |     0.3125  |               0.75 |        0.333333 |   0.3  | balanced       |      nan    |
| logreg_c0.1_balanced                        |   0.465517 |          0.469144 |     0.28125 |               0.8  |        0.333333 |   0.1  | balanced       |      nan    |
| logreg_c3_balanced_flash_gate_0.55_else_ok  |   0.568966 |          0.384824 |     0.71875 |               0.5  |        0        |   3    | balanced       |        0.55 |
| logreg_c1_balanced_flash_gate_0.45_else_ok  |   0.551724 |          0.382246 |     0.625   |               0.6  |        0        |   1    | balanced       |        0.45 |
| logreg_c10_balanced_flash_gate_0.55_else_ok |   0.551724 |          0.378788 |     0.65625 |               0.55 |        0        |  10    | balanced       |        0.55 |
| logreg_c3_balanced_flash_gate_0.45_else_ok  |   0.534483 |          0.373592 |     0.5625  |               0.65 |        0        |   3    | balanced       |        0.45 |
| logreg_c10_balanced                         |   0.534483 |          0.371271 |     0.59375 |               0.6  |        0        |  10    | balanced       |      nan    |
| logreg_c10_plain_flash_gate_0.35_else_ok    |   0.586207 |          0.364316 |     0.875   |               0.3  |        0        |  10    | None           |        0.35 |
| logreg_c10_balanced_flash_gate_0.45_else_ok |   0.517241 |          0.360215 |     0.5625  |               0.6  |        0        |  10    | balanced       |        0.45 |
| logreg_c10_balanced_zs_broken_rescue        |   0.37931  |          0.359024 |     0.375   |               0.35 |        0.5      | nan    | nan            |      nan    |
| logreg_c0.03_balanced_zs_broken_rescue      |   0.362069 |          0.351802 |     0.1875  |               0.55 |        0.666667 | nan    | nan            |      nan    |

## Best classification report

```text
                  precision    recall  f1-score   support

    insulator_ok       0.69      0.28      0.40        32
defect_flashover       0.40      0.85      0.54        20
   defect_broken       1.00      0.33      0.50         6

        accuracy                           0.48        58
       macro avg       0.70      0.49      0.48        58
    weighted avg       0.62      0.48      0.46        58

```