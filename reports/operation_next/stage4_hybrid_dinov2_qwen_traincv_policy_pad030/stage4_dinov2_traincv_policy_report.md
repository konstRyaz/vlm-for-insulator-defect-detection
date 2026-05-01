# Stage 4 DINOv2 + Qwen Train-CV Policy Eval

Feature model: `facebook/dinov2-base`
Classifier: LogisticRegression C=0.03, class_weight=balanced
Selected flashover confidence threshold from train OOF: `0.35`

## Train OOF Threshold Selection
| policy                    |   threshold |   accuracy |   macro_f1 |
|:--------------------------|------------:|-----------:|-----------:|
| flash_lowconf_second_best |        0.35 |   0.638095 |   0.562477 |
| flash_lowconf_second_best |        0.34 |   0.609524 |   0.523443 |
| hard_train_oof            |      nan    |   0.590476 |   0.516962 |
| flash_lowconf_second_best |        0.3  |   0.590476 |   0.516962 |
| flash_lowconf_second_best |        0.32 |   0.590476 |   0.516962 |
| flash_lowconf_second_best |        0.33 |   0.590476 |   0.516962 |
| flash_lowconf_second_best |        0.36 |   0.609524 |   0.421195 |
| flash_lowconf_second_best |        0.38 |   0.628571 |   0.380457 |
| flash_lowconf_second_best |        0.4  |   0.628571 |   0.380457 |
| flash_lowconf_second_best |        0.45 |   0.628571 |   0.380457 |
| flash_lowconf_second_best |        0.5  |   0.628571 |   0.380457 |

## Stage 4 Comparison
| policy                                  |   selected_threshold |   baseline_pipeline_correct |   baseline_pipeline_rate |   hybrid_pipeline_correct |   hybrid_pipeline_rate |   delta_correct |   delta_rate |   hybrid_vlm_correct_on_good |
|:----------------------------------------|---------------------:|----------------------------:|-------------------------:|--------------------------:|-----------------------:|----------------:|-------------:|-----------------------------:|
| dinov2_flash_lowconf_second_best_cv0p35 |                 0.35 |                          23 |                 0.396552 |                        34 |               0.586207 |              11 |    0.189655  |                           34 |
| hard_dinov2                             |                 0.35 |                          23 |                 0.396552 |                        28 |               0.482759 |               5 |    0.0862069 |                           28 |
| qwen_ok_veto_flash_cv0p35               |                 0.35 |                          23 |                 0.396552 |                        27 |               0.465517 |               4 |    0.0689655 |                           27 |

## hard_dinov2

### Metrics
{
  "counts": {
    "gt_objects_total": 58,
    "detector_found_total": 58,
    "good_crop_total": 57,
    "vlm_correct_on_good_crop_total": 28,
    "pipeline_correct_total": 28,
    "ceiling_correct_total": 0
  },
  "rates": {
    "detector_match_rate": 1.0,
    "good_crop_rate_among_matched": 0.9827586206896551,
    "vlm_correct_rate_among_good_pred_crops": 0.49122807017543857,
    "pipeline_correct_rate": 0.4827586206896552,
    "ceiling_correct_rate": 0.0,
    "ceiling_vs_actual_gap": -0.4827586206896552
  },
  "thresholds": {
    "match_iou_threshold": 0.5,
    "good_crop_iou_threshold": 0.7
  }
}

### Error buckets
{
  "counts": {
    "detector_miss": 0,
    "bad_crop_from_detector": 1,
    "vlm_error_on_good_pred_crop": 29,
    "routing_or_filtering_error": 0,
    "correct_pipeline_hit": 28
  },
  "rates": {
    "detector_miss": 0.0,
    "bad_crop_from_detector": 0.017241379310344827,
    "vlm_error_on_good_pred_crop": 0.5,
    "routing_or_filtering_error": 0.0,
    "correct_pipeline_hit": 0.4827586206896552
  }
}

## dinov2_flash_lowconf_second_best_cv0p35

### Metrics
{
  "counts": {
    "gt_objects_total": 58,
    "detector_found_total": 58,
    "good_crop_total": 57,
    "vlm_correct_on_good_crop_total": 34,
    "pipeline_correct_total": 34,
    "ceiling_correct_total": 0
  },
  "rates": {
    "detector_match_rate": 1.0,
    "good_crop_rate_among_matched": 0.9827586206896551,
    "vlm_correct_rate_among_good_pred_crops": 0.5964912280701754,
    "pipeline_correct_rate": 0.5862068965517241,
    "ceiling_correct_rate": 0.0,
    "ceiling_vs_actual_gap": -0.5862068965517241
  },
  "thresholds": {
    "match_iou_threshold": 0.5,
    "good_crop_iou_threshold": 0.7
  }
}

### Error buckets
{
  "counts": {
    "detector_miss": 0,
    "bad_crop_from_detector": 1,
    "vlm_error_on_good_pred_crop": 23,
    "routing_or_filtering_error": 0,
    "correct_pipeline_hit": 34
  },
  "rates": {
    "detector_miss": 0.0,
    "bad_crop_from_detector": 0.017241379310344827,
    "vlm_error_on_good_pred_crop": 0.39655172413793105,
    "routing_or_filtering_error": 0.0,
    "correct_pipeline_hit": 0.5862068965517241
  }
}

## qwen_ok_veto_flash_cv0p35

### Metrics
{
  "counts": {
    "gt_objects_total": 58,
    "detector_found_total": 58,
    "good_crop_total": 57,
    "vlm_correct_on_good_crop_total": 27,
    "pipeline_correct_total": 27,
    "ceiling_correct_total": 0
  },
  "rates": {
    "detector_match_rate": 1.0,
    "good_crop_rate_among_matched": 0.9827586206896551,
    "vlm_correct_rate_among_good_pred_crops": 0.47368421052631576,
    "pipeline_correct_rate": 0.46551724137931033,
    "ceiling_correct_rate": 0.0,
    "ceiling_vs_actual_gap": -0.46551724137931033
  },
  "thresholds": {
    "match_iou_threshold": 0.5,
    "good_crop_iou_threshold": 0.7
  }
}

### Error buckets
{
  "counts": {
    "detector_miss": 0,
    "bad_crop_from_detector": 1,
    "vlm_error_on_good_pred_crop": 30,
    "routing_or_filtering_error": 0,
    "correct_pipeline_hit": 27
  },
  "rates": {
    "detector_miss": 0.0,
    "bad_crop_from_detector": 0.017241379310344827,
    "vlm_error_on_good_pred_crop": 0.5172413793103449,
    "routing_or_filtering_error": 0.0,
    "correct_pipeline_hit": 0.46551724137931033
  }
}