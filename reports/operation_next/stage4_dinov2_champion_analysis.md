# Stage 4 DINOv2 Hybrid Champion Analysis

This report compares the clean Stage 4 Qwen baseline against the DINOv2 + Qwen hybrid branch on the same 58 GT objects. The current champion is `secondbest_cv0p35`: DINOv2 logistic probe with `C=0.03`, balanced class weights, and a train-OOF selected fallback rule that maps low-confidence `defect_flashover` predictions to the second-best DINOv2 class.

## Main Metrics
| run | correct | total | acc | macro3 | recall_insulator_ok | recall_defect_flashover | recall_defect_broken |
|---|---|---|---|---|---|---|---|
| qwen_baseline | 23 | 58 | 0.3966 | 0.3749 | 0.3438 | 0.5000 | 0.3333 |
| hard_dinov2 | 28 | 58 | 0.4828 | 0.4671 | 0.1875 | 0.9500 | 0.5000 |
| qwen_veto_cv035 | 27 | 58 | 0.4655 | 0.4687 | 0.2500 | 0.8000 | 0.5000 |
| champ_secondbest_cv035 | 34 | 58 | 0.5862 | 0.5922 | 0.4688 | 0.7000 | 0.8333 |

## Paired Delta vs Qwen Stage 4 Baseline
| candidate | baseline_correct | candidate_correct | delta_correct | helped | hurt | unchanged_correct | unchanged_wrong | sign_test_p |
|---|---|---|---|---|---|---|---|---|
| hard_dinov2 | 23 | 28 | 5 | 14 | 9 | 14 | 21 | 0.4049 |
| qwen_veto_cv035 | 23 | 27 | 4 | 11 | 7 | 16 | 24 | 0.4807 |
| champ_secondbest_cv035 | 23 | 34 | 11 | 21 | 10 | 13 | 14 | 0.0708 |

Interpretation: the champion improves from `23/58` to `34/58`, with `21` helped objects and `10` hurt objects relative to the Qwen baseline. The exact two-sided sign test on changed cases is about `p=0.0708`; this is a strong practical signal on 58 objects, but still short of a conventional 0.05 threshold, so it should be reported as promising rather than definitive.

## Confusion Matrices
### Qwen baseline
| gt | defect_broken | defect_flashover | insulator_ok | unknown | nan |
|---|---|---|---|---|---|
| defect_broken | 2 | 4 | 0 | 0 | 0 |
| defect_flashover | 3 | 10 | 7 | 0 | 0 |
| insulator_ok | 5 | 12 | 11 | 3 | 1 |

### Hard DINOv2
| gt | defect_broken | defect_flashover | insulator_ok | nan |
|---|---|---|---|---|
| defect_broken | 3 | 2 | 1 | 0 |
| defect_flashover | 1 | 19 | 0 | 0 |
| insulator_ok | 2 | 23 | 6 | 1 |

### Champion second-best cv0.35
| gt | defect_broken | defect_flashover | insulator_ok | nan |
|---|---|---|---|---|
| defect_broken | 5 | 0 | 1 | 0 |
| defect_flashover | 1 | 14 | 5 | 0 |
| insulator_ok | 5 | 11 | 15 | 1 |

## What Improved
The champion is not just a higher single number. It changes the class trade-off. Qwen baseline had better normal-insulator recall than hard DINOv2, but weak defect handling in Stage 4. Hard DINOv2 recovered flashover aggressively and overcalled it. The train-CV second-best fallback reduces that flashover overcall and recovers broken cases, yielding a more balanced profile: `insulator_ok 15/32`, `defect_flashover 14/20`, `defect_broken 5/6`.

## Remaining Failure Pattern
The main remaining failure is still the `insulator_ok` vs `defect_flashover` boundary. Even the champion predicts `defect_flashover` for 11 normal-insulator objects. Conversely, it sends 6 true flashover objects to `insulator_ok` and 1 to `defect_broken`. This means the hybrid branch is substantially better than frozen Qwen, but not yet production-safe without either more data, confidence/review logic, or an expanded test set.

## Helped Cases vs Qwen
| record_id | gt | qwen_pred | hard_pred | champ_pred | matched_pred | match_iou | is_good_crop |
|---|---|---|---|---|---|---|---|
| train_img1_ann1 | defect_broken | defect_flashover | defect_flashover | defect_broken | val_img1_pred1 | 0.7877 | True |
| train_img1_ann2 | defect_broken | defect_flashover | defect_flashover | defect_broken | val_img1_pred14 | 0.9287 | True |
| train_img1_ann4 | insulator_ok | defect_flashover | insulator_ok | insulator_ok | val_img1_pred3 | 0.9592 | True |
| train_img3_ann18 | defect_broken | defect_flashover | defect_broken | defect_broken | val_img2_pred3 | 0.8379 | True |
| train_img5_ann45 | insulator_ok | unknown | defect_flashover | insulator_ok | val_img3_pred3 | 0.9461 | True |
| train_img5_ann46 | insulator_ok | defect_broken | insulator_ok | insulator_ok | val_img3_pred9 | 0.9635 | True |
| val_img9_ann1037 | insulator_ok | defect_flashover | defect_flashover | insulator_ok | val_img4_pred1 | 0.8967 | True |
| val_img10_ann1043 | insulator_ok | defect_flashover | defect_flashover | insulator_ok | val_img5_pred3 | 0.9302 | True |
| val_img10_ann1044 | insulator_ok | defect_flashover | defect_flashover | insulator_ok | val_img5_pred1 | 0.9449 | True |
| val_img17_ann1050 | defect_flashover | insulator_ok | defect_flashover | defect_flashover | val_img6_pred33 | 0.8135 | True |
| val_img17_ann1051 | defect_flashover | defect_broken | defect_flashover | defect_flashover | val_img6_pred32 | 0.9058 | True |
| val_img17_ann1052 | defect_flashover | insulator_ok | defect_flashover | defect_flashover | val_img6_pred46 | 0.8505 | True |
| val_img17_ann1053 | defect_flashover | insulator_ok | defect_flashover | defect_flashover | val_img6_pred18 | 0.8910 | True |
| val_img17_ann1055 | defect_flashover | insulator_ok | defect_flashover | defect_flashover | val_img6_pred24 | 0.9199 | True |
| val_img17_ann1059 | defect_flashover | defect_broken | defect_flashover | defect_flashover | val_img6_pred11 | 0.9085 | True |
| val_img31_ann1063 | insulator_ok | unknown | insulator_ok | insulator_ok | val_img7_pred4 | 0.9377 | True |
| val_img33_ann1067 | insulator_ok | defect_flashover | defect_flashover | insulator_ok | val_img8_pred11 | 0.8845 | True |
| val_img33_ann1069 | defect_flashover | defect_broken | defect_flashover | defect_flashover | val_img8_pred15 | 0.9768 | True |
| val_img33_ann1071 | insulator_ok | defect_broken | defect_flashover | insulator_ok | val_img8_pred2 | 0.9709 | True |
| val_img33_ann1073 | insulator_ok | defect_flashover | defect_flashover | insulator_ok | val_img8_pred4 | 0.9464 | True |
| val_img33_ann1074 | insulator_ok | defect_flashover | defect_flashover | insulator_ok | val_img8_pred10 | 0.8933 | True |

## Hurt Cases vs Qwen
| record_id | gt | qwen_pred | hard_pred | champ_pred | matched_pred | match_iou | is_good_crop |
|---|---|---|---|---|---|---|---|
| train_img1_ann5 | insulator_ok | insulator_ok | defect_flashover | defect_broken | val_img1_pred4 | 0.8870 | True |
| train_img3_ann20 | insulator_ok | insulator_ok | defect_broken | defect_broken | val_img2_pred19 | 0.9512 | True |
| train_img5_ann50 | defect_flashover | defect_flashover | defect_flashover | insulator_ok | val_img3_pred8 | 0.9700 | True |
| val_img9_ann1038 | defect_flashover | defect_flashover | defect_broken | defect_broken | val_img4_pred2 | 0.9048 | True |
| val_img17_ann1045 | insulator_ok | insulator_ok | defect_flashover | defect_flashover | val_img6_pred1 | 0.9253 | True |
| val_img17_ann1047 | insulator_ok | insulator_ok | defect_flashover | defect_flashover | val_img6_pred10 | 0.8827 | True |
| val_img17_ann1048 | insulator_ok | insulator_ok | defect_flashover | defect_flashover | val_img6_pred21 | 0.8783 | True |
| val_img17_ann1057 | insulator_ok | insulator_ok | defect_flashover | defect_flashover | val_img6_pred8 | 0.8215 | True |
| val_img17_ann1058 | insulator_ok | insulator_ok | defect_flashover | defect_flashover | val_img6_pred14 | 0.9330 | True |
| val_img33_ann1068 | defect_flashover | defect_flashover | defect_flashover | insulator_ok | val_img8_pred44 | 0.9059 | True |

## Artifact Index
- `reports/operation_next/analysis_stage4_dinov2_champion_paired_cases.csv`
- `reports/operation_next/analysis_stage4_dinov2_champion_summary.csv`
- `reports/operation_next/analysis_stage4_dinov2_champion_paired_stats.csv`
- `reports/operation_next/analysis_stage4_champion_helped_cases.csv`
- `reports/operation_next/analysis_stage4_champion_hurt_cases.csv`
