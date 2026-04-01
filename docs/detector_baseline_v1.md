# Detector Baseline v1 (Stage 2 Close-Out)

## 1. Purpose / Stage Summary

This document formalizes the Stage 2 detector baseline as a closed and reusable baseline artifact.

The baseline is considered valid `detector_baseline_v1` because:

- training and evaluation completed end-to-end without crashes;
- validation metrics are non-trivial and stable across epochs;
- qualitative outputs and saved artifacts are sufficient for downstream integration;
- detector quality is adequate to unblock Stage 3 (`detector -> VLM`) work.

## 2. Dataset Provenance

- Raw source: IDID raw data on Google Drive (`datasets/idid_raw_v1/Train/...`).
- COCO conversion: `scripts/idid_to_coco.py`.
- Split strategy: train/val split generated from the Train subset.
- Practical packaging: for Kaggle runs, images were copied into COCO split folders.

Detector taxonomy (foreground classes):

- `insulator_ok`
- `defect_flashover`
- `defect_broken`
- `unknown`

Model class count setup:

- `num_classes = 5` for Faster R-CNN (`4` foreground classes + `1` background).

## 3. Training Setup

- Model: `torchvision` Faster R-CNN
- `pretrained: true`
- `epochs: 12`
- `image_size: 640`
- `batch_size: 2`
- `num_workers: 2`
- Training platform: Kaggle GPU (`P100`)

## 4. Quantitative Results

Final validation metrics (`best.pth`):

- `map_50_95 = 0.5663735466926982`
- `map_50 = 0.7596515155328244`
- `map_75 = 0.67626415510823`
- `ap_small = 0.0`
- `ap_medium = -1.0`
- `ap_large = 0.5675354541984097`
- `ar_100 = 0.7384763453064656`

## 5. Training Dynamics

Most metric gains happened during epochs `1-6`; after that the run approached a near-plateau with only marginal improvements.

Interpretation:

- optimization converged normally;
- baseline quality is reproducible enough for reference use;
- additional detector-only tuning is not the current project priority.

## 6. Qualitative Observations

From saved evaluation predictions:

- total predictions: `8398`
- images with predictions: `320`
- mean predictions/image: `26.24`
- median predictions/image: `22`

Class distribution in predictions:

- `insulator_ok`: `4710`
- `defect_flashover`: `2138`
- `defect_broken`: `1253`
- `unknown`: `297`

Operational observation:

- `unknown` behaves as a low-confidence fallback.
- `score_threshold=0.05` is suitable for COCO eval completeness.
- The same threshold is noisy for product/demo-style downstream usage.

## 7. Known Limitations

- No fully normalized local validation GT package is available for perfect forensic TP/FP/FN matching per example.
- `AP_small = 0` and `AP_medium = -1` are not currently treated as pipeline bugs by default.
- No local GPU is available.
- `unknown` remains a semantically weak/suspect class and is handled conservatively.

## 8. Stage Exit Decision

Stage 2 is formally closed as a valid detector baseline stage.

Decision:

- freeze current detector as `detector_baseline_v1`;
- do not open a new heavy detector research loop now;
- proceed to Stage 3 with focus on `detector -> VLM` contract and integration.

## 9. Artifact Manifest

Primary baseline artifacts:

- `outputs/train/detector_baseline/best.pth`
- `outputs/train/detector_baseline/last.pth`
- `outputs/train/detector_baseline/val_predictions_epoch_*.json`
- `outputs/eval/detector_baseline/metrics.json`
- `outputs/eval/detector_baseline/predictions.json`
- `outputs/eval/detector_baseline/vis/`

