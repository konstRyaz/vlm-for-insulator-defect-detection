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

| Metric | tight 0.15 baseline | context 0.50 capped |
| --- | ---: | ---: |
| detector match rate | 1.0000 | 1.0000 |
| good crop rate among matched | 0.9828 | 0.9828 |
| VLM correct rate on good predicted crops | 0.3684 | 0.4035 |
| pipeline correct rate | 0.3621 | 0.3966 |
| Stage 3 ceiling correct rate | 0.4655 | 0.4655 |
| ceiling minus actual gap | 0.1034 | 0.0690 |
| correct pipeline hits | 21/58 | 23/58 |

## Error Trade-Off

| Error pattern | tight 0.15 | context 0.50 capped |
| --- | ---: | ---: |
| `insulator_ok -> defect_flashover` | 15 | 15 |
| `insulator_ok -> defect_broken` | 7 | 3 |
| `defect_flashover -> insulator_ok` | 3 | 8 |
| `defect_flashover -> defect_broken` | 4 | 4 |
| `defect_broken -> defect_flashover` | 3 | 2 |

## Interpretation

The result is a real but narrow improvement. Wider context plus visual-token cap
improves total Stage 4 correctness by two objects and removes the T4 OOM issue.
The main improvement is fewer normal insulators being called broken. The main
unresolved problem remains false flashover on normal insulators, while flashover
recall gets worse.

Current recommendation: keep this as evidence that input context matters, but
run one milder context ablation (`padding_ratio=0.30`, same `max_pixels`) before
freezing a new Stage 4 input strategy.

