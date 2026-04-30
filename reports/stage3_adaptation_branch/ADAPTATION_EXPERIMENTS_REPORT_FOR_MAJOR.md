# Stage 3 Adaptation Experiments Report For Next Planning

Prepared for follow-up planning after the leakage-free Stage 3/4 checkpoint.

## Operational Context

The clean baseline is fixed around the no-crop-path Stage 3 protocol. Historical runs that exposed `crop_path` are diagnostic only. The current reportable reference remains:

- Stage 3 GT-crop baseline: Qwen2.5-VL-3B-Instruct with `qwen_vlm_labels_v1_prompt_v7f_flashover_unclear_to_unknown_nocroppath`.
- Stage 4 best input strategy so far: detector predicted crops with padding `0.30` and Qwen `max_pixels=401408`.
- Main observed bottleneck: semantic classification, especially `insulator_ok` versus `defect_flashover`, not JSON formatting and not detector geometry on the current val slice.

The experiments below were run after the frozen Qwen model sweep showed that larger/newer frozen Qwen variants improve normal-insulator behavior but collapse flashover recall.

## Baseline Anchors

| checkpoint | accuracy / rate | macro-F1 | key note |
|---|---:|---:|---|
| Clean Stage 3 Qwen2.5-VL-3B final | 0.4655 | 0.2882 | Original clean GT-crop ceiling. |
| Clean Stage 3 Qwen2.5-VL-3B rerun in model sweep | 0.4828 | 0.2946 | 28/58; flashover recall 15/20. |
| Qwen2.5-VL-7B frozen | 0.5000 | 0.1556 | Strong normal bias; flashover recall 1/20. |
| Qwen3-VL-4B frozen | 0.5345 | 0.2748 | Strong normal bias; flashover recall 2/20. |
| Stage 4 context pred crop | 0.3966 | n/a | Best actual detector-to-VLM path so far. |

## Experiment 1: CLIP Coarse Benchmark

Notebook: `notebooks/stage3_clip_hybrid_coarse_benchmark_clean.ipynb`  
Kaggle: `kostyaryazanov/notebookd64e91cba0`, version 22.

This experiment tested `openai/clip-vit-base-patch32` as a coarse-only branch. It does not produce `vlm_labels_v1`; it only predicts `coarse_class`.

| method | accuracy | macro-F1 | ok recall | flashover recall | broken recall |
|---|---:|---:|---:|---:|---:|
| zero-shot CLIP | 0.3103 | 0.2858 | 0.3438 | 0.2000 | 0.5000 |
| CLIP linear probe | 0.4483 | 0.3148 | 0.3750 | 0.7000 | 0.0000 |
| CLIP linear probe + unknown threshold | 0.2069 | 0.1739 | 0.0000 | 0.6000 | 0.0000 |

Conclusion: zero-shot CLIP is too weak. A train-balanced linear probe has a useful flashover signal, but loses broken.

## Experiment 2: Exploratory CLIP Hybrid Sweep

Notebook: `notebooks/stage3_clip_hybrid_sweep_clean.ipynb`  
Kaggle: `kostyaryazanov/stage3-clip-hybrid-sweep-clean`, version 1.

This sweep tried multiple logistic-regression settings, flashover thresholds, and a zero-shot broken rescue. It selected by validation macro-F1, so it is not a final claim. It is a hypothesis generator.

Best exploratory method: `logreg_c0.03_balanced`.

| metric | value |
|---|---:|
| accuracy | 0.4828 |
| macro-F1 | 0.4799 |
| ok recall | 0.2813 |
| flashover recall | 0.8500 |
| broken recall | 0.3333 |

Confusion matrix:

```text
GT \ Pred          ok   flashover   broken
insulator_ok       9      23          0
defect_flashover   3      17          0
defect_broken      1       3          2
```

Conclusion: CLIP embeddings contain a strong flashover signal, but naive macro-F1 optimization overcalls flashover on normal insulators.

## Experiment 3: Train-Selected CLIP Hybrid

Notebook: `notebooks/stage3_clip_hybrid_train_selected_clean.ipynb`  
Kaggle: `kostyaryazanov/stage3-clip-hybrid-train-selected-clean`, version 1.

Hyperparameters were selected using 5-fold stratified CV on `train_balanced` only, then clean val was evaluated once.

Selected method: `logreg_c10_balanced`.

| metric | train-CV | clean val |
|---|---:|---:|
| accuracy | 0.7905 | 0.5345 |
| macro-F1 | 0.6912 | 0.3713 |
| ok recall | 0.8250 | 0.5938 |
| flashover recall | 0.8000 | 0.6000 |
| broken recall | 0.6000 | 0.0000 |

Val confusion:

```text
GT \ Pred          ok   flashover   broken
insulator_ok      19      13          0
defect_flashover   8      12          0
defect_broken      4       2          0
```

Conclusion: the train-selected CLIP classifier is the cleanest positive result so far. It beats frozen Qwen on coarse accuracy and macro-F1 in this coarse-only framing, but it still fails `defect_broken`.

## Experiment 4: Broken-Aware CLIP Hybrid

Notebook: `notebooks/stage3_clip_hybrid_broken_aware_clean.ipynb`  
Kaggle: `kostyaryazanov/stage3-clip-hybrid-broken-aware-clean`, version 1.

This added a one-vs-rest broken detector and selected its threshold by train-CV. The selected val result was identical to the train-selected baseline:

| metric | clean val |
|---|---:|
| accuracy | 0.5345 |
| macro-F1 | 0.3713 |
| ok recall | 0.5938 |
| flashover recall | 0.6000 |
| broken recall | 0.0000 |

Conclusion: the simple broken-rescue detector did not generalize to val. Broken remains unresolved.

## Experiment 5: Qwen2.5-VL-3B LoRA Smoke

Notebook: `notebooks/stage3_qwen25vl_3b_lora_smoke_clean.ipynb`  
Kaggle: `kostyaryazanov/notebookd64e91cba0`, version 23.

This was a smoke-first LoRA/QLoRA attempt on `train_balanced`, evaluated on clean val. It completed technically but failed behaviorally.

Observed output:

- raw generations were repeated exclamation marks, e.g. `!!!!!!!!!!!!!!!!...`;
- JSON extraction failed for all 58 samples;
- evaluator fallback turned all predictions into `unknown` / `ambiguous`.

Metrics:

| metric | value |
|---|---:|
| evaluated samples | 58/58 |
| parse success | 0.0000 |
| schema valid | 0.0000 |
| coarse accuracy | 0.0000 |
| coarse macro-F1 | 0.0000 |
| pred ambiguous rate | 1.0000 |

Conclusion: current LoRA notebook is a useful implementation smoke but not a valid model result. The failure likely comes from the crude training format / label masking / generation setup, not from proof that LoRA cannot work.

## Consolidated Findings

1. Frozen model scaling did not solve the bottleneck. Larger/newer Qwen variants shift toward `insulator_ok` and lose defect recall.
2. Broad prompt tuning should remain stopped. It mostly moves the same decision boundary around.
3. CLIP embeddings provide a real discriminative signal, especially for `defect_flashover`.
4. A train-selected CLIP linear probe is currently the strongest clean coarse-only result: `0.5345` accuracy and `0.3713` macro-F1.
5. `defect_broken` is not solved by the simple CLIP branch; it needs either more data, better image features, or a class-specific design.
6. The first LoRA smoke failed at structured generation, so a serious LoRA attempt needs a corrected supervised fine-tuning recipe rather than small parameter tweaking.

## Recommended Next Operation

The next plan should split into two lines.

Line A: build a clean hybrid Stage 3 candidate.

- Use train-selected CLIP linear probe for `coarse_class`.
- Keep Qwen as structured reporter for `visibility`, `visual_evidence_tags`, and text fields.
- Evaluate both Stage 3 GT crops and Stage 4 predicted crops.
- Report it explicitly as a hybrid system, not as a pure VLM.

Line B: repair LoRA training before another long run.

- Mask prompt tokens in labels and train loss only on assistant JSON.
- Reduce target text initially to strict minimal JSON.
- Add a 5-sample overfit check before full val.
- Require parse success on the overfit set before launching a full Kaggle run.

A possible third line is data work: increase clean validation size and add more `defect_broken` examples. Current val has only 6 broken objects, so broken conclusions are unstable.

## Tactical Recommendation

For the next operation, prioritize the hybrid Stage 3 candidate. It already has a clean positive signal. LoRA remains promising but needs implementation repair before it can consume more GPU budget responsibly.
