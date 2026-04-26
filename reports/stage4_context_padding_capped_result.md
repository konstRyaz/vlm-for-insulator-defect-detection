# Stage 4 Context Padding Capped Result

The capped context-padding rerun completed successfully on Kaggle.

## Validity Checks

- run name: `stage4_detector_to_vlm_pred_val_context_pad050_maxpix401k_kaggle`
- prompt: `qwen_vlm_labels_v1_prompt_v7f_flashover_unclear_to_unknown_nocroppath`
- crop padding: `0.50`
- Qwen `max_pixels`: `401408`
- Qwen outputs: `215/215`
- parse success: `215/215`
- schema valid: `215/215`
- backend failures: `0`
- prompt-visible `crop_path`: `false`

## Corrected Metrics

The Kaggle notebook did not include a Stage 3 ceiling input, so the corrected
ceiling comparison was computed locally against the clean v7f Stage 3 run.

| Metric | tight 0.15 baseline | context 0.30 capped | context 0.50 capped |
| --- | ---: | ---: | ---: |
| detector match rate | 1.0000 | 1.0000 | 1.0000 |
| good crop rate among matched | 0.9828 | 0.9828 | 0.9828 |
| VLM correct rate on good predicted crops | 0.3684 | 0.4035 | 0.4035 |
| pipeline correct rate | 0.3621 | 0.3966 | 0.3966 |
| Stage 3 ceiling correct rate | 0.4655 | 0.4655 | 0.4655 |
| ceiling minus actual gap | 0.1034 | 0.0690 | 0.0690 |
| correct pipeline hits | 21/58 | 23/58 | 23/58 |

## Error Trade-Off

| Error pattern | tight 0.15 | context 0.30 capped | context 0.50 capped |
| --- | ---: | ---: | ---: |
| `insulator_ok -> defect_flashover` | 15 | 12 | 15 |
| `insulator_ok -> defect_broken` | 7 | 5 | 3 |
| `defect_flashover -> insulator_ok` | 3 | 7 | 8 |
| `defect_flashover -> defect_broken` | 4 | 3 | 4 |
| `defect_broken -> defect_flashover` | 3 | 4 | 2 |

## Interpretation

The result is a real but narrow improvement. Wider context plus visual-token cap
improves total Stage 4 correctness by two objects and removes the T4 OOM issue.
The `0.30` and `0.50` context variants tie on total accuracy, but `0.30` is the
more balanced candidate: it reduces `insulator_ok -> defect_flashover` from 15
to 12, while preserving more flashover recall than `0.50`.

Current recommendation: keep `padding_ratio=0.30` with `max_pixels=401408` as
the best Stage 4 input-strategy candidate so far. Do not freeze it as final
until one smaller robustness check is considered, but it is currently better
balanced than both the tight crop and the `0.50` context crop.
