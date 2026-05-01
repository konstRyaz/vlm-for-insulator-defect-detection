# Stage 4 DINOv2 + Qwen CV Grid Policy Eval

Feature model: `facebook/dinov2-base`

## Train OOF Selection
|   C | policy      |   threshold |   accuracy |   macro_f1 |
|----:|:------------|------------:|-----------:|-----------:|
| 1   | second_best |        0.38 |   0.72381  |   0.636586 |
| 1   | hard        |      nan    |   0.714286 |   0.633866 |
| 1   | second_best |        0.3  |   0.714286 |   0.633866 |
| 1   | second_best |        0.32 |   0.714286 |   0.633866 |
| 1   | second_best |        0.33 |   0.714286 |   0.633866 |
| 1   | second_best |        0.34 |   0.714286 |   0.633866 |
| 1   | second_best |        0.35 |   0.714286 |   0.633866 |
| 1   | second_best |        0.36 |   0.714286 |   0.633866 |
| 1   | second_best |        0.4  |   0.714286 |   0.615864 |
| 1   | second_best |        0.5  |   0.714286 |   0.594754 |
| 0.3 | second_best |        0.36 |   0.67619  |   0.590476 |
| 0.3 | second_best |        0.35 |   0.67619  |   0.589074 |
| 0.3 | second_best |        0.38 |   0.67619  |   0.581838 |
| 0.3 | second_best |        0.4  |   0.67619  |   0.581838 |
| 0.3 | hard        |      nan    |   0.666667 |   0.579668 |
| 0.3 | second_best |        0.3  |   0.666667 |   0.579668 |
| 0.3 | second_best |        0.32 |   0.666667 |   0.579668 |
| 0.3 | second_best |        0.33 |   0.666667 |   0.579668 |
| 0.3 | second_best |        0.34 |   0.666667 |   0.579668 |
| 1   | second_best |        0.45 |   0.704762 |   0.579243 |

## Stage 4 Comparison
| policy                      |   selected_C | selected_policy   |   selected_threshold |   baseline_pipeline_correct |   baseline_pipeline_rate |   hybrid_pipeline_correct |   hybrid_pipeline_rate |   delta_correct |   delta_rate |   hybrid_vlm_correct_on_good |
|:----------------------------|-------------:|:------------------|---------------------:|----------------------------:|-------------------------:|--------------------------:|-----------------------:|----------------:|-------------:|-----------------------------:|
| cvgrid_C1_second_best_t0p38 |            1 | second_best       |                 0.38 |                          23 |                 0.396552 |                        31 |               0.534483 |               8 |     0.137931 |                           31 |

## cvgrid_C1_second_best_t0p38

### Metrics
{
  "counts": {
    "gt_objects_total": 58,
    "detector_found_total": 58,
    "good_crop_total": 57,
    "vlm_correct_on_good_crop_total": 31,
    "pipeline_correct_total": 31,
    "ceiling_correct_total": 0
  },
  "rates": {
    "detector_match_rate": 1.0,
    "good_crop_rate_among_matched": 0.9827586206896551,
    "vlm_correct_rate_among_good_pred_crops": 0.543859649122807,
    "pipeline_correct_rate": 0.5344827586206896,
    "ceiling_correct_rate": 0.0,
    "ceiling_vs_actual_gap": -0.5344827586206896
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
    "vlm_error_on_good_pred_crop": 26,
    "routing_or_filtering_error": 0,
    "correct_pipeline_hit": 31
  },
  "rates": {
    "detector_miss": 0.0,
    "bad_crop_from_detector": 0.017241379310344827,
    "vlm_error_on_good_pred_crop": 0.4482758620689655,
    "routing_or_filtering_error": 0.0,
    "correct_pipeline_hit": 0.5344827586206896
  }
}