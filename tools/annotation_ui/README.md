# Local Annotation UI (Stage 3 Pilot)

Minimal browser UI for manual annotation of `vlm_labels_v1` pilot JSONL records.

## What it does

- Displays one crop record at a time with readonly metadata.
- Edits EN fields plus tags/visibility/notes.
- Supports previous/next navigation, filters, hotkeys, progress indicator.
- Saves safely to sidecar:
  - input: `vlm_labels_v1_pilot.jsonl`
  - output: `vlm_labels_v1_pilot.annotated.jsonl`
  - if input already ends with `.annotated.jsonl`, it saves back to that same file

## Install

```bash
pip install -r tools/annotation_ui/requirements.txt
```

## Run

```bash
python tools/annotation_ui/app.py \
  --input outputs/stage3_pilot_mini/val/vlm_labels_v1_pilot.jsonl \
  --host 127.0.0.1 \
  --port 8501
```

Open in browser:

`http://127.0.0.1:8501`

## Hotkeys

- `Left`: previous record
- `Right`: next record
- `1`: visibility `clear`
- `2`: visibility `partial`
- `3`: visibility `ambiguous`
- `Ctrl+S`: save now

## Notes

- Source JSONL is not overwritten.
- Sidecar saves are atomic via temp file + replace.
- RU fields are intentionally not written to the main sidecar JSON.
- Record `completed` status is computed from `visibility + visual_evidence_tags + EN text fields`.
- Legacy EN fields remain synchronized:
  - `short_canonical_description = short_canonical_description_en`
  - `report_snippet = report_snippet_en`
