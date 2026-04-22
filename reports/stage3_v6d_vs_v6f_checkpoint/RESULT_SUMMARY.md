# Stage 3 Final Micro-Ablation: stage3_qwen_val_v2_v6d_vs_v6f

- control_version: qwen_vlm_labels_v1_prompt_v6d_balanced_notaglock
- candidate_version: qwen_vlm_labels_v1_prompt_v6f_balanced_notaglock_soft_ambiguous_raise
- selected_prompt: qwen_vlm_labels_v1_prompt_v6d_balanced_notaglock
- recommendation: freeze_v6d_stop_prompt_tuning
- model: Qwen/Qwen2.5-VL-3B-Instruct
- backend: qwen_hf
- gt_jsonl: /kaggle/input/datasets/kostyaryazanov/idid-coco-v3/stage3_regrouped_v2/val/vlm_labels_v1_val_v2.annotated.jsonl
- repo_commit: bf71080