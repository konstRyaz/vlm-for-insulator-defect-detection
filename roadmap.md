# Project Roadmap

This document turns the current project idea into an execution plan with clear deliverables, metrics, and decision gates.

## Project goal

Build an end-to-end system for insulator defect analysis:

`image -> defect detector -> defect description (VLM) -> report`

The project should be strong in two dimensions:

- engineering: reproducible training/evaluation/inference pipeline;
- research: measurable comparison of prompting/input strategies and at least 1-2 justified improvements.

## Status update (2026-04-24)

- Stage 1 is complete for the current baseline path.
- Stage 2 is formally closed with frozen Faster R-CNN `detector_baseline_v1`.
- Historical Stage 3 GT-crop runs are preserved, but prompt-visible `crop_path` leakage was found and clean reruns are required for final reporting.
- Historical Stage 4 detector->VLM runs with prompt-visible `crop_path` are diagnostic only; the clean rerun path is now the active baseline.
- Active execution focus: clean reruns for Stage 3 ceiling, prompt selection, and Stage 4 actual pipeline.
- Current annotation progress is complete for prepared subsets:
  - pilot val: `40/40`
  - train batch: `200/200`
- YOLO remains optional and non-blocking for the current milestone.

Practical note: many sections below remain as planning history. The active execution focus now is leakage-free reruns plus final Stage 4 interpretation.

## Current repo status

What is already present in the repository:

- COCO-style data preparation and validation: `scripts/idid_to_coco.py`, `scripts/prepare_data.py`
- processed split validation reports in `data/processed/reports/`
- training/evaluation/inference pipeline for a detector
- baseline detector implementation based on `torchvision` Faster R-CNN
- Stage 3 contract/spec/docs:
  - `docs/detector_to_vlm_contract.md`
  - `docs/vlm_labels_v1_spec.md`
  - `docs/stage3_gt_bbox_to_vlm_plan.md`
- Stage 3 annotation workflow:
  - GT crop export + bootstrap + validation scripts
  - local annotation UI in `tools/annotation_ui/`

Important practical note:

- the repository already supports Faster R-CNN out of the box and this baseline is already validated;
- current priority is not detector benchmark expansion, but detector-to-VLM integration;
- YOLO can be added later as an optional comparison/appendix branch if time remains.

## Evaluation principles

The project should always separate component quality from end-to-end quality.

For the VLM block, keep three evaluation modes from the beginning:

1. `GT bbox -> VLM`
2. `pred bbox -> VLM`
3. `image -> detector -> VLM -> report`

Why this matters:

- mode 1 estimates the upper bound of the VLM itself;
- mode 2 shows how detector noise affects the VLM;
- mode 3 is the actual product/demo scenario.

## Stage 1 - Data

Status: closed for baseline trajectory

### Goal

Create a stable dataset package `COCO_v1` with train/val/test splits and basic quality control.

### Deliverables

- `COCO_v1` in raw split form
- validated processed dataset in `data/processed/`
- QA reports
- a small `vlm_labels_v1` layer on top of a subset of bounding boxes
- taxonomy document for detector classes and VLM slots

### Definition of done

- `scripts/prepare_data.py` finishes without validation errors
- there is no mismatch between `images[]` and `annotations[]`
- no missing files referenced from COCO JSON
- no duplicate `image_id` or `annotation id`
- bboxes are inside image bounds
- split-level class distribution is known
- at least several saved visual examples exist for manual inspection
- taxonomy is fixed and versioned

### Required artifacts

- `data/processed/reports/validation_report.json`
- `data/processed/reports/train_stats.json`
- `data/processed/reports/val_stats.json`
- `data/processed/reports/test_stats.json` if `test` exists
- `outputs/qa_examples/` or equivalent folder with visual checks
- `docs/taxonomy_v1.md` or similar document if added later

### Minimal VLM label schema

Annotate only a subset first. Keep the schema simple and stable.

Recommended slots:

- `defect_type`
- `severity`
- `visible_parts`
- `confidence_labeler`
- `free_text_description`

Optional later:

- `material`
- `damage_pattern`
- `occlusion`
- `image_quality_issue`

## Stage 2 - Detector baseline

Status: closed as `detector_baseline_v1`

### Goal

Get a working detector with non-zero metrics, understandable failures, and a validated training pipeline.

### Deliverables

- trained baseline checkpoint
- COCO metrics on val/test
- visualization of predictions
- overfit sanity test on 1-5 images
- short error analysis

### Recommended baseline choice

Current project decision:

- keep Faster R-CNN as the official frozen Stage 2 baseline (`detector_baseline_v1`);
- do not start a new heavy detector cycle now;
- treat YOLO as optional later comparison only.

### Definition of done

- training runs end-to-end without crashes
- overfit test on 1-5 images succeeds
- validation `mAP50` is above zero and stable across reruns
- `mAP50-95`, `mAP50`, `AP_small`, and per-class behavior are logged
- prediction visualizations show both successes and typical false positives / false negatives
- at least one checkpoint is selected as the baseline reference model

### Mandatory sanity checks

1. Overfit test on 1-5 images.
2. Training loss decreases.
3. Predictions on training examples become visually close to ground truth.
4. If overfit fails, stop and debug data/pipeline before any larger run.

### Metrics

- `mAP@[.5:.95]`
- `mAP50`
- `AP_small`
- `AR_small`
- per-class AP if more than one defect class is introduced later
- inference latency per image

### Current commands in this repo

Prepare validated data:

```bash
python scripts/prepare_data.py --raw_dir data/raw/idid_coco --out_dir data/processed --dataset coco
```

Train existing baseline:

```bash
python src/train.py +experiment=detector_baseline
```

Evaluate:

```bash
python src/eval.py +experiment=detector_baseline
```

Infer on a folder:

```bash
python src/infer.py +experiment=detector_baseline input_dir=data/processed/val/images output_dir=outputs/infer_idid
```

### Immediate implementation tasks

- add an explicit overfit experiment config
- increase baseline training beyond the current smoke-level settings
- save a short detector error review with FP/FN examples
- if YOLO is added, mirror the same split and evaluation protocol

## Stage 3 - VLM baseline for defect description

### Goal

Make the VLM block independently measurable and usable on detector crops.

### Deliverables

- VLM inference on `crop`
- VLM inference on `crop + context`
- structured output in JSON / slots
- baseline metrics for slot filling quality
- comparison between prompt/input variants

### Input variants to compare

1. `GT crop`
2. `GT crop + context`
3. `pred crop`
4. `pred crop + context`

### Prompt variants to compare

1. zero-shot free-form prompt
2. structured prompt with explicit JSON schema

### Definition of done

- inference runs on a held-out subset without manual intervention
- model output is parsed into a stable schema
- invalid JSON / unparsable outputs are counted explicitly
- slot metrics are computed on the labeled subset
- hallucination / contradiction rate is estimated
- qualitative examples include correct, partially correct, and clearly wrong outputs

### Suggested metrics

- slot accuracy
- macro F1 over categorical slots
- exact match rate for all slots
- invalid JSON rate
- hallucination rate
- contradiction rate against visible evidence or labels

### Recommended first experiment order

1. `GT crop + structured prompt`
2. `GT crop + zero-shot`
3. `GT crop + context + structured prompt`
4. `pred crop + structured prompt`
5. full detector-to-VLM chain

This order gives the cleanest debugging path.

## Stage 4 - Research improvements

### Goal

Show novelty beyond "we connected a detector and a language model".

### Target

Finish with 1-2 focused improvements plus ablations. Do not try to optimize everything.

### Recommended priority

1. `crop` vs `crop + context`
2. zero-shot vs structured prompt
3. ROI marking / SoM style highlighting
4. robustness to degradations
5. LoRA / QLoRA fine-tuning

### Why this order

- the first three are cheaper and easier to interpret;
- robustness gives a strong practical story;
- fine-tuning is valuable, but should come after the base pipeline is stable.

### Definition of done

- each improvement has a clear hypothesis
- each improvement is compared to a fixed baseline
- results are reported in one table with the same evaluation subset
- at least one ablation has a meaningful positive effect
- negative results are documented if the idea does not help

### Candidate experiments

- add a box id or visual marker on the full image and compare against plain crop
- compare raw crop against crop with a controlled amount of surrounding context
- test blur / JPEG compression / downscale robustness
- fine-tune on internal descriptions if enough labels are available

## Stage 5 - Unified system

### Goal

Demonstrate the project as a usable end-to-end system, not only a set of isolated notebooks.

### Deliverables

- `infer_report` CLI or notebook
- per-image JSON report
- Markdown or HTML report
- example input/output folder
- concise reproducibility instructions

### Definition of done

- one command or one notebook cell sequence runs the full pipeline
- outputs contain boxes, detector class, confidence, and VLM description
- outputs are stored in a predictable directory structure
- at least a few showcase examples are curated for the demo

### Recommended report structure

- image path / id
- predicted boxes
- detector scores
- defect class
- VLM slot JSON
- short natural language summary
- optional rendered image with overlays

## Stage 6 - Experiments and thesis write-up

### Goal

Close the course project with a clear experimental story, not only code artifacts.

### Deliverables

- final detector results table
- final VLM results table
- ablation table
- qualitative cases
- written sections for method, experiments, conclusion, and limitations

### Definition of done

- every reported table can be traced to saved outputs
- train/val/test protocol is clearly stated
- component-level and end-to-end evaluation are separated
- limitations are explicit
- claims in the text match actual measured results

### Minimum paper/thesis structure

1. Problem statement
2. Dataset and annotation pipeline
3. Detector baseline
4. VLM baseline
5. Proposed improvements
6. End-to-end system
7. Error analysis and limitations
8. Conclusion

## Stage 7 - Optional extras

Keep this as a backlog, not as a required core milestone.

Possible ideas:

- nicer demo UI
- additional detector family
- active learning loop
- synthetic augmentations
- multilingual report generation

The rule for this stage is simple:

- do not start optional work until stages 2-6 have a minimally complete version.

## Execution order

Recommended practical order from today:

1. Freeze detector baseline artifacts and policy.
2. Prepare `vlm_labels_v1` schema and pilot subset.
3. Build `GT bbox -> VLM` baseline first.
4. Add `pred bbox -> VLM` using frozen detector.
5. Run focused ablations (threshold/top-k/routing, padding/context, unknown handling).
6. Package end-to-end report generation.
7. Finalize tables, figures, and text.

## Decision on YOLO vs Faster R-CNN

Short answer:

- YOLO can still be useful for speed-focused practical comparison later;
- YOLO is not required to proceed with Stage 3 and is not a blocking step now.

Recommended decision rule:

- keep Faster R-CNN baseline frozen for the main project trajectory;
- add YOLO only after Stage 3 baseline is standing and only if time remains.

## Immediate next sprint

The next sprint should produce these concrete outputs:

1. Freeze current annotation snapshot (`pilot 40/40`, `train_200 200/200`) as baseline data version.
2. Implement first `GT bbox -> VLM` baseline runner with structured output.
3. Add baseline evaluator against pilot labels (`visibility`, tag overlap/F1, qualitative error cases).
4. Run first Stage 3 baseline and save reproducible artifacts/results.
5. Prepare transition package for `pred bbox -> VLM` using the frozen detector-to-VLM contract.

If those five items exist, the project has real momentum and the rest becomes much easier to structure.
