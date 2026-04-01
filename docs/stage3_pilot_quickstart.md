# Stage 3 Pilot Quickstart (GT BBox to VLM Labels)

This quickstart creates a small `vlm_labels_v1` pilot set from GT boxes without GPU.

## 1) Export GT crops + manifest

```bash
python scripts/export_vlm_crops.py \
  --coco-json data/processed/val/annotations.json \
  --images-dir data/processed/val/images \
  --output-dir outputs/stage3_gt_crops/val \
  --split val \
  --padding-ratio 0.15 \
  --include-categories defect_flashover defect_broken insulator_ok unknown \
  --manifest-name manifest.jsonl \
  --limit 100
```

Notes:

- If your local dataset is a toy/legacy 1-class variant (`defect`), omit `--include-categories` or adapt values.
- Output artifacts:
  - `manifest.jsonl`
  - `summary.json`
  - `crops/`

## 2) Bootstrap draft `vlm_labels_v1` records

```bash
python scripts/bootstrap_vlm_labels_pilot.py \
  --manifest outputs/stage3_gt_crops/val/manifest.jsonl \
  --output outputs/stage3_gt_crops/val/vlm_labels_v1_pilot.jsonl \
  --limit 100
```

What this does:

- maps manifest records to `vlm_labels_v1` structure;
- pre-fills required text fields with conservative class-based draft text;
- marks `unknown`/`other` as `needs_review=true`.

## 3) Validate pilot labels

```bash
python scripts/validate_vlm_labels_v1.py \
  --input outputs/stage3_gt_crops/val/vlm_labels_v1_pilot.jsonl
```

## 4) Manual annotation pass

Edit pilot records and refine:

- `visual_evidence_tags`
- `visibility`
- `short_canonical_description`
- `report_snippet`
- `annotator_notes`

Guidelines are in:

- `docs/vlm_labels_v1_spec.md`

## 5) Ready for first GT baseline

After pilot labels are manually refined and validated, use them as the initial supervision subset for:

- `GT bbox -> VLM structured output`
- initial prompt and parsing checks
- first slot-level quality estimates before moving to `pred bbox -> VLM`

