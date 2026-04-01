# Detector Baseline Qualitative Analysis

## Scope

This note summarizes Stage 2 qualitative analysis using the saved baseline artifacts:

- `outputs/eval/detector_baseline/metrics.json`
- `outputs/eval/detector_baseline/predictions.json`
- `outputs/eval/detector_baseline/vis/`
- `outputs/train/detector_baseline/train.log`

Important limitation:

- the exact full validation COCO annotations used on Kaggle are not available locally in normalized form;
- because of that, this note does not claim exact per-image TP/FP/FN tagging from local recomputation;
- conclusions below are based on the saved evaluation metrics, raw prediction distribution, and saved qualitative artifacts.

## Final Baseline Result

- `mAP@[.5:.95] = 0.5664`
- `mAP@0.50 = 0.7597`
- `mAP@0.75 = 0.6763`
- `AP_large = 0.5675`
- `AP_small = 0.0`
- `AP_medium = -1.0`
- `AR@100 = 0.7385`

Interpretation:

- the detector is clearly working and does not show pipeline collapse;
- the strongest quality is on large objects;
- small objects are still a weak point;
- the `-1.0` medium metrics most likely mean that the validation split does not contain a meaningful medium-size bucket under COCO area rules.

## Convergence Behavior

Metric trend from `train.log`:

- epoch 1: `0.2559`
- epoch 2: `0.3038`
- epoch 3: `0.4056`
- epoch 4: `0.4745`
- epoch 5: `0.5030`
- epoch 6: `0.5460`
- epoch 7: `0.5559`
- epoch 8: `0.5553`
- epoch 9: `0.5596`
- epoch 10: `0.5613`
- epoch 11: `0.5659`
- epoch 12: `0.5664`

Takeaway:

- training behaved healthily;
- most of the gain came in epochs `1-6`;
- the run began to plateau around epochs `7-9`;
- this is a good sign that the current setup is a valid baseline, not just a lucky one-off.

## Prediction Distribution

From `predictions.json` at the evaluation threshold `0.05`:

- total predictions: `8398`
- images with predictions: `320`
- mean predictions per image: `26.24`
- median predictions per image: `22`

Class distribution:

- `insulator_ok`: `4710` predictions, `56.1%` share
- `defect_flashover`: `2138` predictions, `25.5%` share
- `defect_broken`: `1253` predictions, `14.9%` share
- `unknown`: `297` predictions, `3.5%` share

Confidence behavior:

- `insulator_ok`
  - mean score: `0.6935`
  - median score: `0.8839`
- `defect_flashover`
  - mean score: `0.3514`
  - median score: `0.2190`
- `defect_broken`
  - mean score: `0.3028`
  - median score: `0.1574`
- `unknown`
  - mean score: `0.0996`
  - median score: `0.0733`

Takeaway:

- `insulator_ok` is the most stable and confident class;
- defect classes are much noisier and usually predicted with lower confidence;
- `unknown` behaves mostly like a low-confidence fallback rather than a strong semantic class.

## Threshold Sensitivity

Prediction counts by threshold:

- `0.05`: `8398`
- `0.10`: `6794`
- `0.20`: `5605`
- `0.30`: `4956`
- `0.50`: `4182`
- `0.70`: `3571`
- `0.90`: `2586`

Especially important:

- `unknown` drops from `297` predictions at `0.05` to `11` at `0.30`
- `unknown` disappears completely at `0.50`

Takeaway:

- the saved COCO-eval predictions are intentionally permissive because `score_threshold=0.05`;
- this is good for evaluation completeness, but not good for product-style visualization;
- for demo or downstream detector-to-VLM coupling, a higher threshold like `0.3-0.5` should be tested.

## Qualitative Read Of The Saved Samples

Saved visualization samples:

- `outputs/eval/detector_baseline/vis/000000986d_pred.jpg`
- `outputs/eval/detector_baseline/vis/000001268d_pred.jpg`
- `outputs/eval/detector_baseline/vis/000001268_pred.jpg`
- `outputs/eval/detector_baseline/vis/000006140v_pred.jpg`
- `outputs/eval/detector_baseline/vis/100015_pred.jpg`
- `outputs/eval/detector_baseline/vis/100017h_pred.jpg`
- `outputs/eval/detector_baseline/vis/100017_pred.jpg`
- `outputs/eval/detector_baseline/vis/100020_pred.jpg`

What these saved examples are most useful for:

- illustrating that the baseline is operational on real IDID imagery;
- showing predicted boxes and confidence overlays for report figures;
- selecting a few representative easy and hard cases for the thesis/demo.

What still needs a stricter follow-up if desired:

- exact manual labeling of those samples into TP/FP/FN groups;
- class-confusion analysis on a per-object basis;
- GT-vs-pred overlay figures.

## Main Error Pattern Hypotheses

Based on the metrics and prediction distribution, the most likely dominant error modes are:

- over-prediction in cluttered images when using the very low score threshold;
- confusion between `defect_flashover` and `defect_broken` at moderate confidence;
- instability of the `unknown` class;
- weak performance on small objects;
- many detections per image in dense scenes, which can hurt precision even when recall is decent.

These hypotheses fit the observed pattern:

- good `AR@100` with noisier raw prediction volume;
- strong `insulator_ok` confidence;
- weaker confidence separation for defect classes;
- large-object performance dominating the result.

## Stage 2 Conclusion

Stage 2 can be considered successfully closed as a baseline detector stage because:

- the model trains stably;
- metrics are clearly non-zero and reasonably strong for a first baseline;
- convergence is healthy and understandable;
- saved artifacts are enough for reporting and for the next pipeline stage.

Recommended freeze point:

- keep this detector as `baseline v1`;
- use `best.pth` for downstream VLM experiments;
- do not spend more time on detector tuning until the VLM baseline is standing end-to-end.

## Recommended Next Steps

1. Use the current `best.pth` as the detector input for Stage 3.
2. Keep `metrics.json`, `predictions.json`, and several `vis/*.jpg` files as the official baseline evidence.
3. If needed for the thesis, make one extra note that product-facing inference should probably use a higher score threshold than the COCO-eval default.
