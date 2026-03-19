# MRE: Insulator Defect Detection (PyTorch + COCO + Hydra)

Minimal reproducible project for defect detection on power-line insulators.

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
|   `-- idid_to_coco.py
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
