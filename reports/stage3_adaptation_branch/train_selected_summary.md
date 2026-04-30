# Stage 3 CLIP Hybrid Train-Selected Clean

Hyperparameters were selected using 5-fold stratified cross-validation on `train_balanced` only. The clean val slice was evaluated once after selection.

## Selected method

{'method': 'logreg_c10_balanced', 'C': 10.0, 'class_weight': 'balanced', 'cv_accuracy': 0.7904761904761906, 'cv_macro_f1_3class': 0.6912081128747796, 'cv_ok_recall': 0.825, 'cv_flashover_recall': 0.8, 'cv_broken_recall': 0.5999999999999999}

## Val result

| method              |   C | class_weight   |   cv_accuracy |   cv_macro_f1_3class |   cv_ok_recall |   cv_flashover_recall |   cv_broken_recall |   val_accuracy |   val_macro_f1_3class |   val_ok_recall |   val_flashover_recall |   val_broken_recall |
|:--------------------|----:|:---------------|--------------:|---------------------:|---------------:|----------------------:|-------------------:|---------------:|----------------------:|----------------:|-----------------------:|--------------------:|
| logreg_c10_balanced |  10 | balanced       |      0.790476 |             0.691208 |          0.825 |                   0.8 |                0.6 |       0.534483 |              0.371271 |         0.59375 |                    0.6 |                   0 |

## Val classification report

```text
                  precision    recall  f1-score   support

    insulator_ok       0.61      0.59      0.60        32
defect_flashover       0.44      0.60      0.51        20
   defect_broken       0.00      0.00      0.00         6

        accuracy                           0.53        58
       macro avg       0.35      0.40      0.37        58
    weighted avg       0.49      0.53      0.51        58

```

## Top train-CV candidates

| method                |     C | class_weight   |   cv_accuracy |   cv_macro_f1_3class |   cv_ok_recall |   cv_flashover_recall |   cv_broken_recall |
|:----------------------|------:|:---------------|--------------:|---------------------:|---------------:|----------------------:|-------------------:|
| logreg_c10_balanced   | 10    | balanced       |      0.790476 |             0.691208 |         0.825  |                   0.8 |           0.6      |
| logreg_c3_balanced    |  3    | balanced       |      0.695238 |             0.564076 |         0.725  |                   0.8 |           0.466667 |
| logreg_c0.3_balanced  |  0.3  | balanced       |      0.619048 |             0.538598 |         0.6    |                   0.9 |           0.533333 |
| logreg_c0.01_balanced |  0.01 | balanced       |      0.6      |             0.533631 |         0.5625 |                   0.9 |           0.6      |
| logreg_c1_balanced    |  1    | balanced       |      0.638095 |             0.533242 |         0.65   |                   0.8 |           0.466667 |
| logreg_c0.1_balanced  |  0.1  | balanced       |      0.609524 |             0.529189 |         0.5875 |                   0.9 |           0.533333 |
| logreg_c0.03_balanced |  0.03 | balanced       |      0.6      |             0.522944 |         0.575  |                   0.9 |           0.533333 |
| logreg_c10_plain      | 10    | None           |      0.809524 |             0.474166 |         1      |                   0.3 |           0.133333 |
| logreg_c0.01_plain    |  0.01 | None           |      0.761905 |             0.288288 |         1      |                   0   |           0        |
| logreg_c0.03_plain    |  0.03 | None           |      0.761905 |             0.288288 |         1      |                   0   |           0        |