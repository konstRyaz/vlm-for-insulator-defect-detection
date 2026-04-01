# Stage 3 Entry Plan: GT BBox to VLM

## Why GT BBox First

Stage 3 starts from `GT bbox -> VLM` to separate VLM quality from detector errors.
This gives a clean baseline for structured output quality before noise propagation from predicted boxes.

## Baseline Flow

1. `GT bbox` from COCO annotations
2. crop export with controlled padding/context
3. VLM structured output (schema-constrained)
4. templated sentence generation from structured slots

Target result: stable, analyzable crop-level outputs with low parsing ambiguity.

## Next Step After GT Baseline

Move to `pred bbox -> VLM` using frozen `detector_baseline_v1`.
Keep the same output schema to isolate detector-induced degradation.

## Then Integrate Image-Level Output

1. aggregate region outputs per image
2. apply routing/review/no-defect logic
3. generate final image-level report (`JSON` + textual snippet/markdown)

## Priority Experiments After Baseline

1. `GT bbox` vs `pred bbox` comparison (core robustness gap)
2. threshold/top-k/routing ablation (`0.5` vs `0.3`, max regions, class routing)

Secondary experiments if time remains:

- `unknown` handling ablation (`review` vs `suppress`)
- crop padding/context ablation (`0.0`, `0.15`, `0.30`)

