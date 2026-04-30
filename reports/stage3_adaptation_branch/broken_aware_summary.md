# Stage 3 CLIP Hybrid Broken-Aware Clean

Train-CV selects a main 3-class CLIP linear probe plus a one-vs-rest broken rescue threshold.

## Selected
{'main_C': 10.0, 'main_weight': 'balanced', 'broken_C': 0.03, 'broken_thr': 0.6, 'cv_accuracy': 0.7904761904761906, 'cv_macro_f1_3class': 0.6912081128747796, 'cv_ok_recall': 0.825, 'cv_flashover_recall': 0.8, 'cv_broken_recall': 0.5999999999999999}

## Val
|   main_C | main_weight   |   broken_C |   broken_thr |   cv_accuracy |   cv_macro_f1_3class |   cv_ok_recall |   cv_flashover_recall |   cv_broken_recall |   val_accuracy |   val_macro_f1_3class |   val_ok_recall |   val_flashover_recall |   val_broken_recall |
|---------:|:--------------|-----------:|-------------:|--------------:|---------------------:|---------------:|----------------------:|-------------------:|---------------:|----------------------:|----------------:|-----------------------:|--------------------:|
|       10 | balanced      |       0.03 |          0.6 |      0.790476 |             0.691208 |          0.825 |                   0.8 |                0.6 |       0.534483 |              0.371271 |         0.59375 |                    0.6 |                   0 |

```text
                  precision    recall  f1-score   support

    insulator_ok       0.61      0.59      0.60        32
defect_flashover       0.44      0.60      0.51        20
   defect_broken       0.00      0.00      0.00         6

        accuracy                           0.53        58
       macro avg       0.35      0.40      0.37        58
    weighted avg       0.49      0.53      0.51        58

```

## Top train-CV
|   main_C | main_weight   |   broken_C |   broken_thr |   cv_accuracy |   cv_macro_f1_3class |   cv_ok_recall |   cv_flashover_recall |   cv_broken_recall |
|---------:|:--------------|-----------:|-------------:|--------------:|---------------------:|---------------:|----------------------:|-------------------:|
|       10 | balanced      |       0.03 |          0.6 |      0.790476 |             0.691208 |         0.825  |                   0.8 |           0.6      |
|       10 | balanced      |       0.1  |          0.6 |      0.790476 |             0.691208 |         0.825  |                   0.8 |           0.6      |
|       10 | balanced      |       0.3  |          0.6 |      0.790476 |             0.691208 |         0.825  |                   0.8 |           0.6      |
|       10 | balanced      |       1    |          0.6 |      0.790476 |             0.691208 |         0.825  |                   0.8 |           0.6      |
|       10 | balanced      |       3    |          0.6 |      0.790476 |             0.691208 |         0.825  |                   0.8 |           0.6      |
|       10 | balanced      |      10    |          0.6 |      0.790476 |             0.691208 |         0.825  |                   0.8 |           0.6      |
|       10 | balanced      |      10    |          0.5 |      0.771429 |             0.680405 |         0.7875 |                   0.8 |           0.666667 |
|       10 | balanced      |       3    |          0.5 |      0.752381 |             0.668694 |         0.7625 |                   0.8 |           0.666667 |
|       10 | balanced      |       1    |          0.5 |      0.72381  |             0.64815  |         0.725  |                   0.8 |           0.666667 |
|       10 | balanced      |       0.1  |          0.5 |      0.695238 |             0.628307 |         0.6875 |                   0.8 |           0.666667 |
|       10 | balanced      |       0.3  |          0.5 |      0.695238 |             0.628307 |         0.6875 |                   0.8 |           0.666667 |
|       10 | balanced      |       0.03 |          0.5 |      0.685714 |             0.623706 |         0.675  |                   0.8 |           0.666667 |