# Next research leaderboard

| system | type | accuracy | macro-F1 | notes |
|---|---|---:|---:|---|
| `no_vlm_dinov2_base_logreg` | no-VLM class-only | 0.6552 | 0.6684 | best train-CV-selected feature classifier |
| `stage3_qwen_val_v2_clean_final` | direct VLM GT crop | 0.4655 | 0.2882 | zero-shot structured reporter baseline |
| `stage4_context_pad030_maxpix401k` | direct VLM detector crop | 0.3966 |  | pure Qwen Stage 4 actual baseline |
| `stage4_dinov2_packfix_secondbest035` | hybrid DINOv2+Qwen | 0.5862 | 0.5922 | current hybrid champion; JSON reporter retained |
| `flashover_binary_dinov2_logreg` | accuracy ablation | 0.7115 | 0.7026 | binary ok-vs-flashover diagnostic; train-CV selected |
