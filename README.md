# MRE: Insulator Defect Detection (PyTorch + COCO + Hydra)

Minimal reproducible project for defect detection on power-line insulators.

## Current project status (2026-04-13)

- Stage 2 is formally closed: Faster R-CNN baseline is frozen as `detector_baseline_v1`.
- Stage 3 entry is active and prepared: detector-to-VLM contract, `vlm_labels_v1` schema, and GT-crop tooling are in place.
- Annotation progress is complete for current subsets:
  - pilot val: `40/40`
  - train batch: `200/200`
- Immediate next milestone: first `GT bbox -> VLM -> structured output` baseline run, then move to `pred bbox -> VLM`.
- YOLO remains optional later and is not a current blocking step.

Key docs:

- `docs/detector_baseline_v1.md`
- `docs/detector_to_vlm_contract.md`
- `docs/vlm_labels_v1_spec.md`
- `docs/stage3_gt_bbox_to_vlm_plan.md`
- `docs/stage3_pilot_quickstart.md`
- `docs/stage3_vlm_baseline_quickstart.md`

## Project layout

```text
.
|-- src/
|   |-- configs/
|   |-- train.py
|   |-- eval.py
|   |-- infer.py
|   |-- datasets/
|   |-- model/
|   |-- metrics/
|   |-- logger/
|   `-- utils/
|-- scripts/
|   |-- make_toy_coco.py
|   |-- prepare_data.py
|   |-- idid_to_coco.py
|   `-- export_vlm_crops.py
|-- notebooks/
|   `-- smoke_test.ipynb
|-- requirements.txt
`-- pyproject.toml
```

## Install (Unix/Linux)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data format

Input for `prepare_data.py --dataset coco`:

```text
raw_dir/
|-- train/
|   |-- images/
|   `-- annotations.json
|-- val/
|   |-- images/
|   `-- annotations.json
`-- test/                 # optional
    |-- images/
    `-- annotations.json
```

## Commands

### 1) Build toy COCO dataset

```bash
python scripts/make_toy_coco.py --out_dir data/raw/toy_coco
```

### 2) Validate/prepare data

```bash
python scripts/prepare_data.py --raw_dir data/raw/toy_coco --out_dir data/processed --dataset coco
```

### 3) Train baseline detector

```bash
python src/train.py +experiment=detector_baseline
```

Smoke run (1 epoch, smaller resize):

```bash
python src/train.py +experiment=detector_smoke
python src/eval.py +experiment=detector_smoke
```

### 4) Evaluate checkpoint

```bash
python src/eval.py +experiment=detector_baseline
```

### 5) Run inference on folder

```bash
python src/infer.py +experiment=detector_baseline input_dir=data/processed/val/images output_dir=outputs/infer_toy
```

### 6) Convert IDID labels JSON to COCO train/val

Converts one labels JSON + image directory into COCO split folders.

```bash
python scripts/idid_to_coco.py \
  --input-json /path/to/labels_v1.2.json \
  --images-dir /path/to/images \
  --out-dir data/raw/idid_coco \
  --val-ratio 0.2 \
  --seed 42 \
  --copy-images
```

Optional summary path:

```bash
python scripts/idid_to_coco.py ... --summary-path data/raw/idid_coco/reports/conversion_summary.json
```

### 7) Export GT bbox crops for Stage 3 VLM baseline

```bash
python scripts/export_vlm_crops.py \
  --coco-json data/processed/val/annotations.json \
  --images-dir data/processed/val/images \
  --output-dir outputs/stage3_gt_crops/val \
  --split val \
  --padding-ratio 0.15 \
  --include-categories defect_flashover defect_broken insulator_ok unknown \
  --manifest-name manifest.jsonl \
  --limit 50
```

### 8) Bootstrap and validate `vlm_labels_v1` pilot

```bash
python scripts/bootstrap_vlm_labels_pilot.py \
  --manifest outputs/stage3_gt_crops/val/manifest.jsonl \
  --output outputs/stage3_gt_crops/val/vlm_labels_v1_pilot.jsonl \
  --limit 50

python scripts/validate_vlm_labels_v1.py \
  --input outputs/stage3_gt_crops/val/vlm_labels_v1_pilot.jsonl
```

### 9) Stage 3 Qwen preflight (1 sample)

```bash
python scripts/run_stage3_vlm_baseline.py \
  --config configs/pipeline/stage3_vlm_gt_baseline.yaml \
  --backend-mode qwen_hf \
  --run-id stage3_qwen_preflight_v1 \
  --max-samples 1 \
  --no-resume
```

### 10) Run first Stage 3 VLM baseline (Qwen2.5-VL, smoke)

```bash
python scripts/run_stage3_vlm_baseline.py \
  --config configs/pipeline/stage3_vlm_gt_baseline.yaml \
  --backend-mode qwen_hf \
  --run-id stage3_qwen_smoke_v1 \
  --max-samples 8 \
  --no-resume
```

### 11) Run full Stage 3 `val_v2` baseline (Qwen2.5-VL)

```bash
python scripts/run_stage3_vlm_baseline.py \
  --config configs/pipeline/stage3_vlm_gt_baseline.yaml \
  --backend-mode qwen_hf \
  --run-id stage3_qwen_val_v2 \
  --no-resume
```

### 12) Evaluate Stage 3 VLM baseline outputs

```bash
python scripts/eval_stage3_vlm_baseline.py \
  --run-dir outputs/stage3_vlm_baseline_runs/stage3_qwen_smoke_v1 \
  --ground-truth-jsonl outputs/stage3_regrouped_v2/val/vlm_labels_v1_val_v2.annotated.jsonl
```

Notebook-oriented Stage 3 quickstart (Kaggle primary, Colab fallback):

- `docs/stage3_vlm_baseline_quickstart.md`

## What is saved

- Hydra resolved config: `outputs/.../.hydra/`
- Checkpoints: `outputs/train/<run_name>/{last.pth,best.pth,epoch_XXX.pth}`
- Evaluation metrics: `outputs/eval/.../metrics.json`
- Predictions (COCO detections json): `outputs/.../predictions.json`
- Visualizations: `outputs/.../vis/`

## Repository policy

- This repository is code-first.
- Local data and run artifacts are intentionally not versioned in Git:
  - `data/`
  - `outputs/`
  - `.venv/`

## Notes

- Runtime Hydra configs are in `src/configs` (`train.py`, `eval.py`, `infer.py` use `config_path="configs"` relative to `src/`).
- Baseline model: `torchvision` Faster R-CNN.
- Resize mode defaults to aspect-preserving letterbox (`resize_mode=pad`).
- `resize_mode=stretch` is available via config override.
- `metrics.json` contains all 12 COCOeval bbox stats.
- Console output highlights `mAP@[.5:.95]`, `AP_small`, and `AR_small`.
