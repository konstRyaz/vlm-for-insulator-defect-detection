#!/usr/bin/env python3
"""Audit train/validation split and write a protocol note.

This does not train any model. It documents the role of train/val for the next
research stage and checks for obvious overlaps/leakage risks.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

SUSPICIOUS_TOKENS = [
    "insulator_ok",
    "defect_flashover",
    "defect_broken",
    "flashover",
    "broken",
    "normal",
    "ok",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit train/val split protocol.")
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--val-jsonl", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/next_research/protocol"))
    parser.add_argument("--labels", default="insulator_ok,defect_flashover,defect_broken,unknown")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(obj)
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def nonempty_str(row: Dict[str, Any], key: str) -> str:
    value = row.get(key)
    return str(value).strip() if value is not None else ""


def basename_or_empty(value: str) -> str:
    return Path(value).name if value else ""


def bbox_key(row: Dict[str, Any]) -> str:
    image_id = nonempty_str(row, "image_id")
    bbox = row.get("bbox_xywh")
    if isinstance(bbox, list) and len(bbox) >= 4:
        rounded = [round(float(x), 1) for x in bbox[:4]]
        return f"{image_id}:{rounded}"
    return image_id


def overlap(a: Iterable[str], b: Iterable[str]) -> List[str]:
    aa = {x for x in a if x}
    bb = {x for x in b if x}
    return sorted(aa & bb)


def path_suspicious(path: str) -> List[str]:
    low = path.lower().replace("\\", "/")
    hits = []
    for token in SUSPICIOUS_TOKENS:
        # Mark exact class-like tokens in file/folder names. Common token "ok" is noisy;
        # still report it as a warning, not as a hard failure.
        pattern = re.escape(token.lower())
        if re.search(rf"(^|[/_\-.]){pattern}($|[/_\-.])", low):
            hits.append(token)
    return hits


def class_distribution(rows: List[Dict[str, Any]], labels: Sequence[str], split: str) -> List[Dict[str, Any]]:
    counts = Counter(nonempty_str(row, "coarse_class") for row in rows)
    out = []
    total = len(rows)
    for label in labels:
        c = counts.get(label, 0)
        out.append({"split": split, "label": label, "count": c, "rate": c / total if total else 0.0})
    extras = sorted(set(counts) - set(labels))
    for label in extras:
        c = counts[label]
        out.append({"split": split, "label": label, "count": c, "rate": c / total if total else 0.0})
    return out


def main() -> None:
    args = parse_args()
    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train = read_jsonl(args.train_jsonl)
    val = read_jsonl(args.val_jsonl)

    train_record_ids = [nonempty_str(r, "record_id") for r in train]
    val_record_ids = [nonempty_str(r, "record_id") for r in val]
    train_crop_paths = [nonempty_str(r, "crop_path") for r in train]
    val_crop_paths = [nonempty_str(r, "crop_path") for r in val]
    train_crop_basenames = [basename_or_empty(p) for p in train_crop_paths]
    val_crop_basenames = [basename_or_empty(p) for p in val_crop_paths]
    train_image_ids = [nonempty_str(r, "image_id") for r in train]
    val_image_ids = [nonempty_str(r, "image_id") for r in val]
    train_bbox_keys = [bbox_key(r) for r in train]
    val_bbox_keys = [bbox_key(r) for r in val]

    suspicious_rows: List[Dict[str, Any]] = []
    for split_name, rows in [("train", train), ("val", val)]:
        for row in rows:
            p = nonempty_str(row, "crop_path")
            hits = path_suspicious(p)
            if hits:
                suspicious_rows.append({
                    "split": split_name,
                    "record_id": nonempty_str(row, "record_id"),
                    "coarse_class": nonempty_str(row, "coarse_class"),
                    "crop_path": p,
                    "suspicious_tokens": ";".join(hits),
                    "note": "Do not expose this path to VLM prompts. It may still be fine as a local file path.",
                })

    audit = {
        "train_jsonl": str(args.train_jsonl),
        "val_jsonl": str(args.val_jsonl),
        "counts": {
            "train_total": len(train),
            "val_total": len(val),
            "train_unique_record_ids": len(set(train_record_ids)),
            "val_unique_record_ids": len(set(val_record_ids)),
        },
        "overlaps": {
            "record_id": overlap(train_record_ids, val_record_ids),
            "crop_path_exact": overlap(train_crop_paths, val_crop_paths),
            "crop_basename": overlap(train_crop_basenames, val_crop_basenames),
            "image_id": overlap(train_image_ids, val_image_ids),
            "image_id_bbox_rounded": overlap(train_bbox_keys, val_bbox_keys),
        },
        "warnings": {
            "suspicious_crop_path_rows": len(suspicious_rows),
            "class_path_warning": "Class-like tokens in crop_path are not a failure by themselves, but crop_path must not be shown to VLM prompts.",
        },
    }
    write_json(args.out_dir / "train_val_overlap_audit.json", audit)
    write_csv(args.out_dir / "train_val_distribution.csv", class_distribution(train, labels, "train") + class_distribution(val, labels, "val"))
    write_csv(args.out_dir / "suspicious_crop_path_tokens.csv", suspicious_rows)

    overlap_counts = {k: len(v) for k, v in audit["overlaps"].items()}
    hard_failure = overlap_counts["record_id"] > 0 or overlap_counts["crop_path_exact"] > 0
    lines = [
        "# Train/validation protocol note",
        "",
        "## Summary",
        "",
        f"- Train rows: `{len(train)}`",
        f"- Validation rows: `{len(val)}`",
        f"- Record-id overlap: `{overlap_counts['record_id']}`",
        f"- Exact crop-path overlap: `{overlap_counts['crop_path_exact']}`",
        f"- Crop basename overlap: `{overlap_counts['crop_basename']}`",
        f"- Image-id overlap: `{overlap_counts['image_id']}`",
        f"- Rounded image+bbox overlap: `{overlap_counts['image_id_bbox_rounded']}`",
        f"- Suspicious crop-path token rows: `{len(suspicious_rows)}`",
        "",
        "## Methodology statement",
        "",
        "Direct VLM baselines are zero-shot/inference-only: Qwen/InternVL/LLaVA-like models are not trained on the train split.",
        "The train split is used for prompt/config selection and for trainable branches: feature-based classifiers, hybrid classifier policy and LoRA/SFT experiments.",
        "The validation split is used for final evaluation only. If a prompt/threshold was changed after inspecting validation errors, that result must be marked as diagnostic and revalidated with train-CV or a fresh split.",
        "",
        "## Leakage rule",
        "",
        "Do not pass `crop_path`, class-coded folder names or label-like filenames into VLM prompts. Local file paths may contain class tokens for storage, but model-visible text must remain no-crop-path/no-label.",
        "",
        "## Gate",
        "",
        "PASS" if not hard_failure else "FAIL: record_id or exact crop path overlap detected",
    ]
    (args.out_dir / "train_val_protocol.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {args.out_dir / 'train_val_protocol.md'}")


if __name__ == "__main__":
    main()
