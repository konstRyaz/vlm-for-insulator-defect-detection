# Stage 3 CLIP Hybrid Train-Selected Result

Run source: Kaggle kernel `kostyaryazanov/stage3-clip-hybrid-train-selected-clean`, version 1.

This experiment selected CLIP linear-probe hyperparameters using 5-fold stratified cross-validation on `train_balanced` only. The clean val slice was evaluated once after model selection.

## Selected Method

`logreg_c10_balanced`

Train-CV metrics:

- accuracy: `0.7905`
- macro-F1: `0.6912`
- insulator_ok recall: `0.8250`
- defect_flashover recall: `0.8000`
- defect_broken recall: `0.6000`

Clean val metrics:

- accuracy: `0.5345`
- macro-F1: `0.3713`
- insulator_ok recall: `0.5938`
- defect_flashover recall: `0.6000`
- defect_broken recall: `0.0000`

Val confusion matrix:

```text
,insulator_ok,defect_flashover,defect_broken
insulator_ok,19,13,0
defect_flashover,8,12,0
defect_broken,4,2,0
```

## Interpretation

The train-selected CLIP linear probe is a cleaner signal than the exploratory val sweep. It beats the frozen Qwen2.5-VL-3B clean rerun on raw accuracy and macro-F1, and it keeps a more balanced `insulator_ok` / `defect_flashover` trade-off than either the over-flashover exploratory sweep or the normal-biased larger Qwen variants.

The limitation is clear: the selected model never predicts `defect_broken` on val. That makes it unsuitable as a complete replacement for the structured Qwen Stage 3 path. It is, however, a credible hybrid candidate for the `insulator_ok` versus `defect_flashover` boundary.

## Decision

Keep this as the first clean hybrid baseline. The next useful branch is either:

1. integrate this coarse classifier as an auxiliary decision component and let Qwen generate the remaining structured fields, or
2. train a broken-aware variant with explicit class balancing or a one-vs-rest broken detector.
