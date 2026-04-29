# Stage 3 Adaptation Branch Plan

This branch follows the frozen VLM model sweep and Qwen3 prompt-repair result.

The current evidence is:

- Qwen2.5-VL-3B keeps the best defect recall balance but has weak `insulator_ok` recall.
- Qwen2.5-VL-7B and Qwen3-VL-4B improve `insulator_ok` but collapse most `defect_flashover` recall.
- Prompt-only repair for Qwen3-VL-4B did not recover enough flashover recall.
- The model oracle across 3B, 7B, and Qwen3-4B is much higher than any single model, so there is complementary signal.

## Active Experiments

### 1. Hybrid coarse classifier plus Qwen reporter

Notebook:

`notebooks/stage3_clip_hybrid_coarse_benchmark_clean.ipynb`

This trains/evaluates a CLIP-style discriminative coarse classifier:

- zero-shot CLIP text/image ranking;
- CLIP image embeddings + linear probe on clean `train_balanced`;
- clean `val_v2` evaluation.

This is a coarse-only experiment. If it improves the `insulator_ok` versus `defect_flashover` boundary, the intended system is:

`crop -> coarse classifier -> Qwen structured reporter`

The Qwen reporter would still produce `visual_evidence_tags`, `visibility`, and report snippets.

### 2. TL-CLIP coarse-only benchmark

TL-CLIP is relevant as a power-domain CLIP-style model. Current public search found the paper/arXiv record, but no confirmed runnable public weights/code path.

Until weights are available, TL-CLIP should be treated as related work and as a drop-in target for the CLIP-style benchmark interface, not as a completed experiment.

### 3. Qwen2.5-VL-3B LoRA/QLoRA

Notebook:

`notebooks/stage3_qwen25vl_3b_lora_smoke_clean.ipynb`

This is a smoke-first supervised adaptation experiment:

- train on clean `train_balanced` only;
- evaluate on clean `val_v2` only;
- preserve `vlm_labels_v1` output fields;
- keep no-crop-path prompt protocol.

Because the dataset is small, this branch should be judged cautiously. A useful run should improve coarse macro-F1 and not merely overfit one class.

## Execution Order

1. Run the CLIP/hybrid coarse benchmark first. It is cheaper and tests the strongest non-generative hypothesis.
2. If CLIP-style coarse classification is promising, consider a Stage 3 hybrid integration path.
3. Run Qwen LoRA smoke after that. It is heavier and more failure-prone.
4. Add TL-CLIP only if usable weights/code are attached as a reproducible input.

## Sources

- TL-CLIP arXiv: https://arxiv.org/abs/2411.11370
- TL-CLIP summary page: https://www.catalyzex.com/paper/tl-clip-a-power-specific-multimodal-pre
