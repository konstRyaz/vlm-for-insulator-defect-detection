# Briefing for the Next Research Plan

Date: 2026-05-01  
Repository: `vlm-for-insulator-defect-detection`  
Current operation: leakage-free Stage 3/4 evaluation and hybrid improvement after the Major's previous plan.

## 1. Executive Summary

The main research conclusion has become clearer. The detector and crop geometry are not the dominant bottleneck on the current clean validation slice. The dominant bottleneck is crop-level semantic classification, especially the boundary between `insulator_ok` and `defect_flashover`, with `defect_broken` limited by very small support.

The strongest current result is a hybrid Stage 4 system:

`detector pred crop -> DINOv2 coarse classifier -> Qwen structured reporter`

The best confirmed variant is:

`stage4_dinov2_packfix_secondbest035`

It uses:

- feature backbone: `facebook/dinov2-base`
- classifier: `LogisticRegression(C=0.03, class_weight=balanced)`
- policy: if DINOv2 top class is low-confidence `defect_flashover`, replace it with the second-best DINOv2 class
- threshold: `0.35`, selected by train OOF-CV, not by validation tuning
- reporter: frozen Qwen2.5-VL-3B structured outputs, with only `coarse_class` overridden
- Stage 4 crop policy: detector predicted boxes, context padding `0.30`, max pixels `401408`

Best Stage 4 result:

| system | correct | rate | macro-F1 3-class | ok recall | flashover recall | broken recall |
|---|---:|---:|---:|---:|---:|---:|
| Qwen Stage 4 baseline | 23/58 | 0.3966 | 0.3749 | 0.3438 | 0.5000 | 0.3333 |
| hard DINOv2 hybrid | 28/58 | 0.4828 | 0.4671 | 0.1875 | 0.9500 | 0.5000 |
| DINOv2 train-CV second-best fallback | 34/58 | 0.5862 | 0.5922 | 0.4688 | 0.7000 | 0.8333 |

Paired comparison against the Qwen Stage 4 baseline:

- delta: `+11/58` objects
- helped: `21`
- hurt: `10`
- unchanged correct: `13`
- unchanged wrong: `14`
- exact sign-test on changed cases: approximately `p=0.0708`

This is a strong practical improvement on a small validation slice, but not yet a statistically definitive deployment claim.

## 2. Clean Protocol Status

Leakage cleanup remains central. Earlier crop-path leakage made historical high scores diagnostic-only. Current clean results avoid exposing `crop_path` or class-coded path tokens to the VLM. The final Qwen prompt path is `_nocroppath`.

Current clean baselines:

| result | score | note |
|---|---:|---|
| Stage 3 Qwen GT-crop clean baseline | 0.4655 acc | frozen Qwen reporter baseline |
| Stage 4 Qwen context baseline | 23/58 = 0.3966 | actual detector-to-VLM path |
| Stage 3 DINOv2 train-CV policy | 32/58 = 0.5517 | coarse-only GT-crop control |
| Stage 4 DINOv2 train-CV hybrid | 34/58 = 0.5862 | current main result |

Important caveat: Stage 4 hybrid is currently a hard field-replacement system. Qwen reporter fields remain Qwen-generated, while `coarse_class` comes from DINOv2. This can create semantic inconsistency between class and text/tags. It is acceptable as a research hybrid baseline, but must be reported honestly.

## 3. What Was Tested

### Frozen Qwen/VLM direction

Frozen Qwen2.5-VL-3B gave stable JSON but weak semantics. Larger/newer Qwen-family frozen models shifted class bias rather than solving the task:

- Qwen2.5-VL-7B: higher normal bias, poor flashover recall.
- Qwen3-VL-4B: higher raw accuracy in one run, but weak flashover recall.
- Prompt repair did not recover the needed defect boundary reliably.

Conclusion: broad frozen VLM scaling is not the most promising next path on this dataset.

### Prompt tuning

Prompt-only work improved format stability and helped calibrate visibility earlier, but clean semantic metrics saturated. The final useful prompt is `qwen_vlm_labels_v1_prompt_v7f_flashover_unclear_to_unknown_nocroppath`. Further broad prompt sweeps are unlikely to be the best use of GPU time.

### CLIP/DINO/SigLIP discriminative branch

A feature-backbone sweep showed that discriminative visual features carry stronger coarse-class signal than frozen Qwen alone. DINOv2 was strongest among tested coarse-only feature backbones.

Backbone sweep key results:

| model | acc | macro-F1 | ok recall | flashover recall | broken recall |
|---|---:|---:|---:|---:|---:|
| CLIP B/32 | 0.5690 | 0.4609 | 0.7500 | 0.4000 | 0.1667 |
| CLIP L/14 | 0.5172 | 0.4739 | 0.6563 | 0.3000 | 0.5000 |
| SigLIP B/16 | 0.4828 | 0.5300 | 0.3750 | 0.6000 | 0.6667 |
| DINOv2 base | 0.6552 | 0.6684 | 0.6563 | 0.6500 | 0.6667 |

The later train-CV second-best policy was more conservative and gave Stage 3 `32/58` but better class balance and Stage 4 transfer.

### LoRA/QLoRA

LoRA is not disproven, but the current recipe failed the semantic gate:

- first attempt: repeated punctuation / invalid JSON
- repaired parse gate: `5/5` valid JSON on smoke
- semantic gate: failed, collapsed to `defect_flashover`

Per the Major's acceptance rules, full LoRA val was correctly skipped.

## 4. Champion Error Analysis

The champion improves the full Stage 4 path from `23/58` to `34/58`. The improvement is not just from flashover overcalling. The second-best fallback reduces the worst hard-DINOv2 behavior and recovers broken defects.

Champion confusion matrix:

| GT | pred broken | pred flashover | pred ok | no pred |
|---|---:|---:|---:|---:|
| defect_broken | 5 | 0 | 1 | 0 |
| defect_flashover | 1 | 14 | 5 | 0 |
| insulator_ok | 5 | 11 | 15 | 1 |

Remaining failure mode:

- `insulator_ok -> defect_flashover`: 11 cases remain
- `defect_flashover -> insulator_ok`: 5 cases remain
- `insulator_ok -> defect_broken`: 5 cases remain

Interpretation: the current hybrid is much better balanced than Qwen or hard-DINOv2, but it still lacks robust fine-grained separation of normal dark/complex insulator crops from flashover-like visual evidence.

## 5. Operation Plan Sync

The Major's previous plan predicted that the best next path would be a discriminative coarse classifier plus Qwen reporter. That prediction is now supported.

Status against the plan:

| Phase | Status | Notes |
|---|---|---|
| Phase A: registry + paired eval | partially done | registry, leaderboard, helped/hurt, sign test done; bootstrap CI still needed |
| Phase B: hybrid Stage 3/4 | strongly advanced | current best result is the DINOv2 hybrid |
| Phase C: backbone sweep | advanced | DINOv2 found as strongest tested feature backbone |
| Phase D: broken branch | partially addressed | champion reaches 5/6 broken on Stage 4, but data audit still needed |
| Phase E: LoRA repair | smoke only | stopped correctly after semantic collapse |
| Phase F/G: VLM/domain zoo | diagnostic only | no urgent GPU priority unless a clearly runnable domain model appears |
| Phase H: dataset expansion | not started | needed for strong final claims |

## 6. Risks and Limitations

- Validation set has only 58 objects. One object is 1.72 percentage points.
- The paired sign-test is promising but not conventionally significant at 0.05.
- The final system is a hybrid, not a pure VLM.
- Qwen text/tags may disagree with the DINOv2-overridden `coarse_class`.
- Broken recall is encouraging but based on only 6 validation examples.
- Final claims should be framed as a reproducible evidence-decomposition and hybrid baseline, not production-grade accuracy.

## 7. Recommended Next Plan Questions for the Major

1. Should the next operation prioritize statistical/reporting hardening or more GPU experiments?
2. Should we formalize the hybrid as the main system, or keep it as a diagnostic branch while pursuing LoRA/domain VLMs?
3. Should we add a classifier-conditioned reporter step so Qwen regenerates text/tags consistent with DINOv2's class?
4. Should the next data effort focus on more `defect_broken` examples, more normal dark false positives, or a larger final test split?
5. Should we implement a review/abstention layer, measuring coverage-risk rather than raw accuracy only?

## 8. Recommended Immediate Actions

1. Add formal paired-statistics script with bootstrap CI.
2. Add formal no-leak audit command for prompts, run summaries, manifests, and model-visible text.
3. Build a visual helped/hurt review for the champion.
4. Test classifier-conditioned Qwen reporter only after the above reporting tools are in place.
5. Start planning a larger clean dev/test split; without more data, result interpretation will remain limited.

## 9. Files Included in This Package

The archive contains:

- this briefing report;
- operation registry and leaderboard;
- Stage 4 DINOv2 champion analysis;
- Stage 3 DINOv2 train-CV control results;
- Stage 4 champion artifacts, including predictions and case tables;
- Kaggle notebooks used for the latest DINOv2 hybrid experiments;
- relevant Stage 3/Stage 4 scripts/configs/docs;
- the Major's previous operation plan for continuity.
