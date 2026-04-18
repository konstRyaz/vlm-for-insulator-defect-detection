# Stage 3 VLM Baseline Quickstart (Qwen2.5-VL, Kaggle/Colab)

This Stage 3 baseline remains:

`annotated GT crop dataset -> VLM -> parsed prediction -> vlm_labels_v1 mapping -> validation -> evaluation`

Recommended runtime is cloud notebook GPU:

- primary: Kaggle
- fallback: Colab

## 1) Baseline contract

Config: `configs/pipeline/stage3_vlm_gt_baseline.yaml`

Prompt assets:

- `configs/pipeline/prompts/stage3_vlm_system_v1.txt`
- `configs/pipeline/prompts/stage3_vlm_user_v1.txt`

Prediction contract mode: `reduced_subset_v1`

Model-predicted core fields:

- `coarse_class`
- `visual_evidence_tags`
- `visibility`
- `short_canonical_description_en`
- `report_snippet_en`

Optional debug model field:

- `annotator_notes`

Pipeline-derived fields:

- `needs_review = (visibility == "ambiguous")`
- `short_canonical_description = short_canonical_description_en`
- `report_snippet = report_snippet_en`

Pipeline-copied metadata:

- `record_id`, `image_id`, `box_id`, `source`, `split`, `bbox_xywh`
- `crop_path`, `image_path`, `score`, `category_name`, `label_version`

## 2) Input dataset

Default config points to:

- `outputs/stage3_regrouped_v2/val/vlm_labels_v1_val_v2.annotated.jsonl`

Expected record fields include at least:

- `record_id`, `image_id`, `box_id`, `source`, `split`, `bbox_xywh`, `crop_path`

## 3) Kaggle (recommended)

Enable GPU in notebook settings, then run from repo root.

Install/upgrade dependencies:

```bash
pip install -q -U transformers accelerate
# Optional but recommended for official Qwen multimodal preprocessing path:
pip install -q -U qwen-vl-utils
```

Preflight run (1 sample, checks model load + one end-to-end record):

```bash
python scripts/run_stage3_vlm_baseline.py \
  --config configs/pipeline/stage3_vlm_gt_baseline.yaml \
  --backend-mode qwen_hf \
  --run-id stage3_qwen_preflight_v1 \
  --max-samples 1 \
  --no-resume
```

Tiny smoke-run (5-8 samples):

```bash
python scripts/run_stage3_vlm_baseline.py \
  --config configs/pipeline/stage3_vlm_gt_baseline.yaml \
  --backend-mode qwen_hf \
  --run-id stage3_qwen_smoke_v1 \
  --max-samples 8 \
  --no-resume
```

Full `val_v2` run:

```bash
python scripts/run_stage3_vlm_baseline.py \
  --config configs/pipeline/stage3_vlm_gt_baseline.yaml \
  --backend-mode qwen_hf \
  --run-id stage3_qwen_val_v2 \
  --no-resume
```

Optional targeted subset (record IDs):

```bash
python scripts/run_stage3_vlm_baseline.py \
  --config configs/pipeline/stage3_vlm_gt_baseline.yaml \
  --backend-mode qwen_hf \
  --sample-ids-file sample_ids.txt \
  --run-id stage3_qwen_targeted_v1 \
  --no-resume
```

## 4) Colab (fallback)

Use the same commands as Kaggle after:

1. mounting data/repo (for example from Google Drive),
2. `cd` into repo root,
3. installing dependencies.

Example setup:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
cd /content/drive/MyDrive/vlm-for-insulator-defect-detection
pip install -q -U transformers accelerate
pip install -q -U qwen-vl-utils
```

Then run the same smoke/full commands shown in Kaggle section.

## 5) Validate and evaluate

Validate predicted `vlm_labels_v1` JSONL:

```bash
python scripts/validate_vlm_labels_v1.py \
  --input outputs/stage3_vlm_baseline_runs/<run_id>/predictions_vlm_labels_v1.jsonl
```

Evaluate:

```bash
python scripts/eval_stage3_vlm_baseline.py \
  --run-dir outputs/stage3_vlm_baseline_runs/<run_id> \
  --ground-truth-jsonl outputs/stage3_regrouped_v2/val/vlm_labels_v1_val_v2.annotated.jsonl
```

## 6) Output artifacts

Run directory:

- `outputs/stage3_vlm_baseline_runs/<run_id>/sample_results.jsonl`
- `outputs/stage3_vlm_baseline_runs/<run_id>/raw_responses.jsonl`
- `outputs/stage3_vlm_baseline_runs/<run_id>/parsed_predictions.jsonl`
- `outputs/stage3_vlm_baseline_runs/<run_id>/predictions_vlm_labels_v1.jsonl`
- `outputs/stage3_vlm_baseline_runs/<run_id>/failures.jsonl`
- `outputs/stage3_vlm_baseline_runs/<run_id>/run_summary.json`
- `outputs/stage3_vlm_baseline_runs/<run_id>/config_snapshot.json`

Evaluation directory (`<run_dir>/eval` by default):

- `metrics.json`
- `summary.md`
- `review_table.csv`
- `failures.jsonl`
- `confusion_coarse_class.csv`
- `confusion_visibility.csv`
