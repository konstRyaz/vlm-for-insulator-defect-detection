# Flashover binary ablation

Targeted diagnostic for the main error boundary: `insulator_ok` vs `defect_flashover`.

| run | accuracy | macro-F1 | normal recall | flashover recall |
|---|---:|---:|---:|---:|
| `non_vlm_dinov2_binary_logreg` | 0.7115 | 0.7026 | 0.7188 | 0.7000 |
| `non_vlm_clip_b32_binary_logreg` | 0.6154 | 0.6061 | 0.6250 | 0.6000 |

Best binary diagnostic: `non_vlm_dinov2_binary_logreg` with accuracy `0.7115` and flashover recall `0.7000`.

This is diagnostic, not a final Stage 4 policy by itself. Any threshold/review rule must be selected by train-CV before being claimed as clean final.
