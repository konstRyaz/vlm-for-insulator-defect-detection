# Final Operation Report - Clean VLM/Hybrid Research Checkpoint

Date: 2026-05-02

## Executive Summary

The current research checkpoint is leakage-free and reproducible. The project moved from a frozen Qwen Stage 3 VLM baseline into a detector-to-VLM Stage 4 pipeline, then into model comparison, LoRA repair, and hybrid classifier+reporter experiments.

The strongest current result is not a larger frozen VLM and not LoRA. It is a hybrid Stage 4 system:

`detector predicted crop -> DINOv2 coarse classifier policy -> Qwen structured reporter fields`

The confirmed Stage 4 champion is `stage4_dinov2_packfix_secondbest035`:

- Qwen Stage 4 context baseline: `23/58 = 0.3966` pipeline-correct objects.
- DINOv2+Qwen champion: `34/58 = 0.5862` pipeline-correct objects.
- Absolute improvement: `+11/58 = +0.1897`.
- Paired helped/hurt: `21 helped`, `10 hurt`.
- Sign-test p-value: `0.0708`.
- Bootstrap paired delta CI: `[0.0000, 0.3793]`.

This is a strong practical signal on the current 58-object validation slice, but it is not yet a statistically definitive deployment claim.

## Fixed Clean Protocol

The reportable experiments use the no-leak Stage 3/Stage 4 protocol:

- no `crop_path` or class-coded filename tokens are passed to the VLM prompt;
- Stage 3 is the GT-crop VLM ceiling;
- Stage 4 is the detector-predicted-crop actual pipeline;
- `vlm_labels_v1` remains the structured output contract;
- Qwen2.5-VL-3B remains the canonical structured reporter baseline;
- detector baseline remains frozen.

A no-leak audit over the current champion artifacts found `0` prompt-visible leakage hits.

## Main Baselines

| checkpoint | result | interpretation |
|---|---:|---|
| Stage 3 Qwen clean final | `27/58 = 0.4655` | original clean GT-crop ceiling |
| Stage 3 Qwen model-sweep control | `28/58 = 0.4828` | small reproducibility band |
| Stage 4 Qwen context pad030 | `23/58 = 0.3966` | best pure Qwen detector-to-VLM path |
| Stage 4 DINOv2+Qwen champion | `34/58 = 0.5862` | current best research result |

## Frozen VLM Comparison

A frozen VLM comparison was run under the clean Stage 3 protocol. The tested non-Qwen models did not produce a better structured reporter.

| model | parse/schema | acc | macro-F1 | decision |
|---|---:|---:|---:|---|
| Qwen2.5-VL-3B control | `1.0 / 1.0` | `0.4828` | `0.2946` | baseline anchor |
| InternVL3-2B base | `1.0 / 1.0` | `0.5517` | `0.2853` | higher raw acc, not promoted |
| InternVL3-2B defect-recall | `1.0 / 1.0` | `0.3966` | `0.2255` | overcalled defects |
| InternVL3-2B balanced | `1.0 / 1.0` | `0.5000` | `0.2316` | low defect recall |
| LLaVA-OneVision 0.5B | `0.7931 / 0.2414` | `0.1207` | `0.0609` | failed schema/semantics |
| SmolVLM2 2.2B | `0.0 / 0.0` | `0.0` | `0.0` | failed parse |
| SmolVLM2 500M | `0.6034 / 0.6034` | `0.3276` | `0.1134` | class collapse |
| Phi-3.5-Vision | not run | n/a | n/a | generic pipeline incompatible |

Conclusion: broad frozen VLM comparison is closed for now. No non-Qwen frozen VLM is promoted to Stage 4.

## Specialized Model Audit

The plan named three domain-specific VLM candidates: TL-CLIP, PowerGPT, and Power-LLaVA.

Current status:

| candidate | result |
|---|---|
| TL-CLIP | no public runnable weights/code found; future coarse-only benchmark if released |
| PowerGPT | no public runnable weights/API/code found; related work only |
| Power-LLaVA | no official/provenance-clear runnable release found; related work only |

No GPU benchmark was run for these models because there was no reproducible model to run.

## LoRA/SFT Adaptation

The repaired Qwen2.5-VL-3B LoRA/SFT smoke completed and reached full validation. It fixed the earlier invalid-output failure, but did not improve the task.

Corrected metrics:

- parse/schema: `1.0000 / 1.0000`;
- coarse accuracy: `0.5172`;
- coarse macro-F1: `0.1579`;
- flashover recall: `1/20`;
- broken recall: `0/6`.

Decision: do not promote to Stage 4. This is a negative adaptation checkpoint: the model learned stable JSON but collapsed toward `insulator_ok`.

## Hybrid Branch

The hybrid branch is the main positive result. DINOv2 features plus a train-CV selected policy improved Stage 4 from `23/58` to `34/58`.

Champion class profile:

- `insulator_ok`: `15/32` recall;
- `defect_flashover`: `14/20` recall;
- `defect_broken`: `5/6` recall.

The main remaining error is still `insulator_ok` vs `defect_flashover`. The champion improves class balance, but still predicts flashover for 11 normal-insulator objects.

## Scientific Contribution

The project contribution is not a claim of production-grade accuracy. The contribution is a clean, reproducible object-level evaluation protocol for structured-output VLM analysis of insulator defects:

1. separate GT-crop VLM ceiling from detector-to-VLM actual quality;
2. detect and remove prompt-visible path leakage;
3. compare frozen VLMs, LoRA adaptation, and hybrid classifier+reporter variants under the same clean split;
4. attribute errors to detector miss, crop quality, and VLM semantic classification;
5. show that a discriminative visual backbone can improve the actual pipeline more than broad prompt tuning or frozen VLM swaps.

## Limitations

The validation slice has 58 GT objects. One object changes accuracy by about 1.7 percentage points. Results should be treated as reproducible baseline and error-decomposition evidence, not final deployment performance.

The hybrid system can produce inconsistent text/tags because only `coarse_class` is overridden while Qwen reporter fields are retained. This is acceptable for research decomposition, but should be addressed before any user-facing report layer.

## Recommended Next Work

1. Freeze `stage4_dinov2_packfix_secondbest035` as the current main Stage 4 research result.
2. Stop broad frozen VLM and broad prompt sweeps for now.
3. If improving metrics is still required, target one of these precise branches:
   - larger clean validation/test set;
   - DINOv2 hybrid confidence/review policy for ok-vs-flashover errors;
   - class-balanced LoRA/SFT with a stronger objective, not naive next-token SFT;
   - future TL-CLIP/PowerGPT/Power-LLaVA only if reproducible releases appear.
4. For writing, emphasize the protocol, leakage removal, and error decomposition rather than high absolute accuracy.
