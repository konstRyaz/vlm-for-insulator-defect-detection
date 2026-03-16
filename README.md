# MRE: Insulator Defect Detection (PyTorch + COCO + Hydra)

Minimal reproducible project for defect detection on power-line insulators.

## Project layout

```text
.
├── src/
│   ├── configs/
│   ├── train.py
│   ├── eval.py
│   ├── infer.py
│   ├── datasets/
│   ├── model/
│   ├── metrics/
│   ├── logger/
│   └── utils/
├── scripts/
│   ├── prepare_data.py
│   └── make_toy_coco.py
├── notebooks/
│   └── smoke_test.ipynb
├── requirements.txt
└── pyproject.toml
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data format

Input for `prepare_data.py --dataset coco`:

```text
raw_dir/
├── train/
│   ├── images/
│   └── annotations.json
├── val/
│   ├── images/
│   └── annotations.json
└── test/                 # optional
    ├── images/
    └── annotations.json
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

## What is saved

- Hydra resolved config: `outputs/.../.hydra/`
- Checkpoints: `outputs/train/<run_name>/{last.pth,best.pth,epoch_XXX.pth}`
- Evaluation metrics: `outputs/eval/.../metrics.json`
- Predictions (COCO detections json): `outputs/.../predictions.json`
- Visualizations: `outputs/.../vis/`

## Notes

- Baseline model: `torchvision` Faster R-CNN.
- Resize mode defaults to aspect-preserving letterbox (`resize_mode=pad`).
- `resize_mode=stretch` is available via config override.
- `metrics.json` contains all 12 COCOeval bbox stats.
- Console output highlights `mAP@[.5:.95]`, `AP_small`, and `AR_small`.
