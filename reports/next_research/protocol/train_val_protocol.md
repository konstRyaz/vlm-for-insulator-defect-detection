# Train/validation protocol note

## Summary

- Train rows: `105`
- Validation rows: `58`
- Record-id overlap: `0`
- Exact crop-path overlap: `0`
- Crop basename overlap: `0`
- Image-id overlap: `0`
- Rounded image+bbox overlap: `0`
- Suspicious crop-path token rows: `163`

## Methodology statement

Direct VLM baselines are zero-shot/inference-only: Qwen/InternVL/LLaVA-like models are not trained on the train split.
The train split is used for prompt/config selection and for trainable branches: feature-based classifiers, hybrid classifier policy and LoRA/SFT experiments.
The validation split is used for final evaluation only. If a prompt/threshold was changed after inspecting validation errors, that result must be marked as diagnostic and revalidated with train-CV or a fresh split.

## Leakage rule

Do not pass `crop_path`, class-coded folder names or label-like filenames into VLM prompts. Local file paths may contain class tokens for storage, but model-visible text must remain no-crop-path/no-label.

## Gate

PASS