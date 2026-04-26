# Final Research Summary

This is the current leakage-free checkpoint for the insulator defect pipeline. The final path is:

`image -> frozen detector -> predicted crop -> Qwen2.5-VL -> vlm_labels_v1 structured output -> evaluation`

The old prompt-visible `crop_path` runs are preserved only as diagnostic history. Final Stage 3 and Stage 4 numbers below use the no-leak prompt family and should be treated as the reportable results.

## Fixed Setup

Detector baseline is frozen as `detector_baseline_v1` with Faster R-CNN. The VLM baseline is Qwen/Qwen2.5-VL-3B-Instruct. The clean Stage 3 prompt is `qwen_vlm_labels_v1_prompt_v7f_flashover_unclear_to_unknown_nocroppath`, which avoids passing crop path tokens or class-like filenames into the prompt.

Stage 3 is the GT-crop ceiling. Stage 4 is the actual detector-to-VLM path on predicted crops. This separation is the main experimental design choice: it lets us distinguish detector/crop loss from VLM semantic loss.

## Main Results

| Component | Metric | Value |
| --- | ---: | ---: |
| Stage 2 detector | mAP@[.5:.95] | 0.5664 |
| Stage 2 detector | mAP@0.50 | 0.7597 |
| Stage 2 detector | AR@100 | 0.7385 |
| Stage 3 GT crop VLM | coarse accuracy | 0.4655 |
| Stage 3 GT crop VLM | coarse macro-F1 | 0.2882 |
| Stage 3 GT crop VLM | visibility accuracy | 0.8448 |
| Stage 3 GT crop VLM | visibility macro-F1 | 0.5356 |
| Stage 3 GT crop VLM | parse/schema success | 1.0000 / 1.0000 |
| Stage 4 tight pred crop | pipeline correct rate | 0.3621 |
| Stage 4 context pred crop | pipeline correct rate | 0.3966 |
| Stage 4 context pred crop | ceiling gap | 0.0690 |

The best current Stage 4 input candidate is predicted crop context padding `0.30` with Qwen `max_pixels=401408`. It improves correctness from `21/58` to `23/58` objects compared with the tight predicted crop, while reducing the Stage 3-to-Stage 4 gap from `0.1034` to `0.0690`.

## What The Decomposition Shows

The detector is not the primary bottleneck in the current validation slice. Stage 4 matching found all `58/58` GT objects, and `57/58` matched crops passed the good-crop IoU threshold. The main remaining bottleneck is semantic classification by the frozen VLM on the crop.

Context helps, but it is not a universal fix. The `0.30` context crop helped 10 objects and hurt 8. It improved normal-insulator behavior, but flashover-vs-normal confusion remains the central failure mode.

## Important Error Pattern

The clearest unresolved boundary is `insulator_ok` versus `defect_flashover`. Even after prompt tuning, the model often treats dark local visual cues as flashover-like evidence, while some true flashover objects still get predicted as normal or another defect class. This is why the current result is scientifically useful but not yet production-grade.

## Final Artifacts

The compact final package is in `reports/final_stage4_package/`. It includes:

- `final_metrics_table.csv`
- copied Stage 4 context comparison CSV tables
- copied Stage 4 comparison plots
- `stage4_context_comparison_report.md`

The source comparison report remains at `reports/stage4_context_comparison_final/report.md`.

## Research Contribution

The project now has a reproducible open-source VLM evaluation path for insulator defect crops, plus a clean ceiling-vs-actual decomposition. The main contribution is not claiming high absolute accuracy; it is showing where the end-to-end detector-to-VLM pipeline loses quality and which input-context change closes part of that gap.

This gives a defensible research narrative: a frozen detector and frozen open-source VLM can be connected in a structured-output pipeline, but the main limiting factor is fine-grained visual semantics rather than JSON validity or detector geometry.

## Limitations

The final validation slice has 58 GT objects, so every object changes accuracy by about 1.7 percentage points. The VLM is frozen and not fine-tuned. The detector is also frozen. The results are therefore best read as a reproducible baseline and error decomposition, not as a final deployment model.

## Suggested Next Step

Stop broad prompt tuning for Stage 3. Further gains are more likely from one of three directions: more labeled clean validation data, a stronger or fine-tuned VLM, or a small decision/review layer targeted at the flashover-vs-normal boundary.
