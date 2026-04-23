# Stage 4 Error Review (local eval with ceiling)

This note summarizes where Stage 4 loses quality on val after detector crops are passed into VLM.

## Main numbers

- GT objects: 58
- Detector match rate: 1.0000
- Good crop rate among matched: 0.9828
- VLM correct rate among good pred crops: 0.5789
- Pipeline correct rate (actual): 0.5690
- Ceiling correct rate (Stage 3 GT crop): 0.9310
- Ceiling - actual gap: 0.3621

## Where the drop comes from

- Total VLM errors on good predicted crops: 24
- Of them, additional drop vs ceiling (ceiling was correct): 21
- Of them, already wrong at ceiling too: 3

## Pattern breakdown (GT -> Pred) on 24 VLM errors

| GT class | Pred class | Count |
|---|---:|---:|
| defect_flashover | insulator_ok | 11 |
| insulator_ok | defect_flashover | 6 |
| insulator_ok | defect_broken | 3 |
| defect_flashover | defect_broken | 2 |
| defect_broken | insulator_ok | 1 |
| defect_broken | defect_flashover | 1 |

Key pattern: the biggest failure is `defect_flashover -> insulator_ok` (11 cases).

## Visibility in error subset

- Visibility mismatches inside these 24 errors: 3
- Mismatch record_ids: train_img1_ann3, train_img3_ann20, val_img33_ann1072

Conclusion: current Stage 4 loss is mostly coarse-class confusion, not visibility calibration.

## Important caution

The current Stage 4 run appears to leak detector class metadata into the VLM prompt.

- On good predicted crops, `detector category == VLM coarse class` in `56/57` cases.
- On the same set, `detector category == GT` only in `34/57` cases.
- The user prompt includes `crop_path`, and predicted crop paths encode detector class folders such as `crops/val/insulator_ok/...`.

This means the current Stage 4 metrics are useful as a diagnostic checkpoint, but not yet as a clean estimate of visual-only `pred crop -> VLM` quality.

## Class-wise VLM accuracy on good predicted crops

| GT class | Samples | Correct rate |
|---|---:|---:|
| defect_broken | 6 | 0.6667 |
| defect_flashover | 20 | 0.3500 |
| insulator_ok | 31 | 0.7097 |

The weakest class is `defect_flashover` (0.3500 on good predicted crops).

## Recommended next pass

1. First remove `crop_path` from the Stage 4 prompt context so the VLM cannot read detector class tokens from folder names.
2. Re-run Stage 4 cleanly and recompute the same visual package.
3. Only after that, if needed, do a narrow prompt pass focused on `flashover vs insulator_ok`.
