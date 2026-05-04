# Structured output comparison

The draft manual rubric below is an initial structured-field review. It is useful for triage, but visual human review is still recommended before publication claims about description quality.

| run | n | coarse | visibility | tag Jaccard | desc present | draft usefulness | draft consistency |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen_stage3_clean | 58 | 0.4655 | 0.8448 | 0.1822 | 1.0000 | 1.4483 | 1.4138 |
| qwen_stage4_context_pad030_matched | 58 | 0.3966 | 0.7931 | 0.1443 | 1.0000 | 1.3966 | 1.3793 |
| hybrid_dinov2_qwen_champion_matched | 58 | 0.5862 | 0.7931 | 0.1443 | 1.0000 | 1.5862 | 1.2759 |
| internvl3_2b_base | 58 | 0.5517 | 0.7931 | 0.0330 | 1.0000 | 1.4828 | 0.7931 |

Main reading: direct and hybrid VLM systems keep structured fields present, but tag overlap remains modest. The hybrid improves class accuracy on matched Stage 4 objects while retaining Qwen reporter fields; however, class override can create text/tag inconsistency and should be handled by a future reporter-regeneration or consistency-check layer.
