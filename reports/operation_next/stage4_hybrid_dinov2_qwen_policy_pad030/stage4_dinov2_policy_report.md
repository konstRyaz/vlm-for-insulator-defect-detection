# Stage 4 DINOv2 + Qwen Policy Probe

Feature model: `facebook/dinov2-base`
Classifier: LogisticRegression C=0.03, class_weight=balanced

## Comparison
| policy                   |   baseline_pipeline_correct |   baseline_pipeline_rate |   hybrid_pipeline_correct |   hybrid_pipeline_rate |   delta_correct |   delta_rate |   hybrid_vlm_correct_on_good |
|:-------------------------|----------------------------:|-------------------------:|--------------------------:|-----------------------:|----------------:|-------------:|-----------------------------:|
| qwen_ok_veto_flash_lt034 |                          23 |                 0.396552 |                        29 |               0.5      |               6 |    0.103448  |                           29 |
| hard_dinov2              |                          23 |                 0.396552 |                        28 |               0.482759 |               5 |    0.0862069 |                           28 |

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

## qwen_ok_veto_flash_lt034

### Metrics
{
  "counts": {
    "gt_objects_total": 58,
    "detector_found_total": 58,
    "good_crop_total": 57,
    "vlm_correct_on_good_crop_total": 29,
    "pipeline_correct_total": 29,
    "ceiling_correct_total": 0
  },
  "rates": {
    "detector_match_rate": 1.0,
    "good_crop_rate_among_matched": 0.9827586206896551,
    "vlm_correct_rate_among_good_pred_crops": 0.5087719298245614,
    "pipeline_correct_rate": 0.5,
    "ceiling_correct_rate": 0.0,
    "ceiling_vs_actual_gap": -0.5
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
    "vlm_error_on_good_pred_crop": 28,
    "routing_or_filtering_error": 0,
    "correct_pipeline_hit": 29
  },
  "rates": {
    "detector_miss": 0.0,
    "bad_crop_from_detector": 0.017241379310344827,
    "vlm_error_on_good_pred_crop": 0.4827586206896552,
    "routing_or_filtering_error": 0.0,
    "correct_pipeline_hit": 0.5
  }
}