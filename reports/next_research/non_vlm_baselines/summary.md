# Non-VLM baseline summary

Feature classifiers were selected by train-CV only and evaluated once on clean `val_v2`.

## Leaderboard

| run_id | kind | model_id | classifier | C | class_weight | cv_macro_f1_mean | accuracy | macro_f1 | recall_insulator_ok | recall_defect_flashover | recall_defect_broken |
|---|---|---|---|---|---|---|---|---|---|---|---|
| non_vlm_dinov2_base_logreg | hf_auto | facebook/dinov2-base | logreg | 0.0300 | balanced | 0.8220 | 0.6552 | 0.6684 | 0.6562 | 0.6500 | 0.6667 |
| non_vlm_efficientnet_b0_logreg | timm | efficientnet_b0.ra_in1k | logreg | 0.1000 | balanced | 0.6608 | 0.6207 | 0.6062 | 0.6875 | 0.5000 | 0.6667 |
| non_vlm_convnext_tiny_logreg | timm | convnext_tiny.fb_in1k | logreg | 1.0000 | balanced | 0.5627 | 0.6552 | 0.5902 | 0.7188 | 0.6000 | 0.5000 |
| non_vlm_siglip_b16_224_logreg | hf_auto | google/siglip-base-patch16-224 | logreg | 0.0300 | balanced | 0.7615 | 0.4828 | 0.5300 | 0.3750 | 0.6000 | 0.6667 |
| non_vlm_clip_l14_logreg | hf_auto | openai/clip-vit-large-patch14 | logreg | 3.0000 | none | 0.7403 | 0.5172 | 0.4739 | 0.6562 | 0.3000 | 0.5000 |
| non_vlm_clip_b32_logreg | hf_auto | openai/clip-vit-base-patch32 | logreg | 0.0300 | balanced | 0.8048 | 0.5690 | 0.4609 | 0.7500 | 0.4000 | 0.1667 |
| non_vlm_resnet50_logreg | timm | resnet50.a1_in1k | logreg | 1.0000 | balanced | 0.6478 | 0.4483 | 0.4580 | 0.4375 | 0.4500 | 0.5000 |

## Main result

Best no-VLM baseline: `non_vlm_dinov2_base_logreg` with accuracy `0.6552` and macro-F1 `0.6684`.

This confirms that visual-feature classifiers are stronger than the direct frozen VLM for raw coarse classification on the current crop-level validation slice. The VLM value should therefore be evaluated as structured reporting, not only as a class predictor.
