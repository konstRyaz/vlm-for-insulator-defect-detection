# Stage 4 DINOv2 + Qwen Reporter Light Eval

Feature model: `facebook/dinov2-base`
Classifier: LogisticRegression C=0.03, class_weight=balanced

## Comparison
|   baseline_pipeline_correct |   baseline_pipeline_rate |   hybrid_pipeline_correct |   hybrid_pipeline_rate |   delta_correct |   delta_rate |   hybrid_vlm_correct_on_good |
|----------------------------:|-------------------------:|--------------------------:|-----------------------:|----------------:|-------------:|-----------------------------:|
|                          23 |                 0.396552 |                        28 |               0.482759 |               5 |    0.0862069 |                           28 |

## Hybrid metrics
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

## Error buckets
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