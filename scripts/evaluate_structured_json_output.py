#!/usr/bin/env python3
"""Evaluate structured JSON output beyond coarse-class accuracy.

This script is complementary to existing eval_stage3/eval_stage4 scripts. It focuses on
whether the VLM/reporter output is useful as a structured report:
- schema/field availability;
- class/visibility/tag agreement;
- automatic description sanity checks;
- manual review template and summary.

Manual rubric columns are generated as empty columns. After a human fills scores 0/1/2,
rerun the script with --manual-review-csv to aggregate them.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

COARSE_LABELS = ["insulator_ok", "defect_flashover", "defect_broken", "unknown", "other"]
VISIBILITY_LABELS = ["clear", "partial", "ambiguous"]
MANUAL_SCORE_FIELDS = [
    "manual_tag_score",
    "manual_description_relevance",
    "manual_visual_evidence_score",
    "manual_hallucination_score",
    "manual_usefulness_score",
    "manual_class_text_consistency",
]
GENERIC_DESCRIPTION_PATTERNS = [
    r"cannot determine",
    r"unclear",
    r"not sure",
    r"appears to be an insulator",
    r"visible insulator",
    r"image shows",
    r"no obvious",
]
CLASS_TAG_HINTS = {
    "defect_flashover": {"flashover", "dark", "burn", "arc", "trace", "surface", "char", "black"},
    "defect_broken": {"broken", "crack", "missing", "fracture", "chip", "fragment", "damage"},
    "insulator_ok": {"intact", "normal", "clean", "regular", "no_damage", "no defect"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate structured JSON reporter fields.")
    parser.add_argument("--gt-jsonl", required=True, type=Path)
    parser.add_argument("--pred-jsonl", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/next_research/structured_output_eval"))
    parser.add_argument("--sample-results-jsonl", type=Path, default=None, help="Optional raw run sample_results with parse/schema info.")
    parser.add_argument("--manual-review-csv", type=Path, default=None, help="Optional filled manual review CSV to aggregate.")
    parser.add_argument("--max-review-rows", type=int, default=0, help="0 means all rows.")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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


def by_record_id(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        rid = row.get("record_id")
        if isinstance(rid, str) and rid.strip():
            out[rid] = row
    return out


def normalize_tags(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip().lower())
    return sorted(set(out))


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    aa, bb = set(a), set(b)
    if not aa and not bb:
        return 1.0
    union = aa | bb
    return len(aa & bb) / len(union) if union else 0.0


def text_fields(row: Dict[str, Any]) -> Tuple[str, str, str]:
    short = row.get("short_canonical_description") or row.get("short_canonical_description_en") or row.get("short_canonical_description_ru") or ""
    snippet = row.get("report_snippet") or row.get("report_snippet_en") or row.get("report_snippet_ru") or ""
    joined = f"{short} {snippet}".strip()
    return str(short), str(snippet), str(joined)


def generic_description_flag(text: str) -> bool:
    low = text.lower().strip()
    if len(low) < 20:
        return True
    return any(re.search(pat, low) for pat in GENERIC_DESCRIPTION_PATTERNS)


def class_text_consistency_auto(pred_class: str, text: str, tags: Sequence[str]) -> str:
    low = text.lower() + " " + " ".join(tags).lower()
    hints = CLASS_TAG_HINTS.get(pred_class, set())
    if not hints:
        return "unknown"
    hit = any(h in low for h in hints)
    if pred_class == "insulator_ok":
        bad_defect = any(h in low for h in CLASS_TAG_HINTS["defect_flashover"] | CLASS_TAG_HINTS["defect_broken"])
        if bad_defect:
            return "contradiction_possible"
    return "consistent_hint" if hit else "no_class_specific_hint"


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "1", "yes"}:
            return True
        if low in {"false", "0", "no"}:
            return False
    return None


def read_manual_scores(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("record_id")
            if rid:
                out[rid] = row
    return out


def score_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def aggregate_manual(review_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for field in MANUAL_SCORE_FIELDS:
        vals = [score_value(row.get(field)) for row in review_rows]
        vals2 = [v for v in vals if v is not None]
        out[field] = {
            "n_scored": len(vals2),
            "mean": safe_div(sum(vals2), len(vals2)),
            "score_counts": dict(Counter(str(int(v)) if v is not None and abs(v - int(v)) < 1e-9 else str(v) for v in vals2)),
        }
    return out


def markdown_table(rows: List[Dict[str, Any]], fields: Sequence[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        vals = []
        for f in fields:
            v = row.get(f, "")
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v).replace("\n", " ")[:200])
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_dir = args.out_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    gt_by_id = by_record_id(read_jsonl(args.gt_jsonl))
    pred_by_id = by_record_id(read_jsonl(args.pred_jsonl))
    sample_by_id: Dict[str, Dict[str, Any]] = {}
    if args.sample_results_jsonl and args.sample_results_jsonl.exists():
        sample_by_id = by_record_id(read_jsonl(args.sample_results_jsonl))

    eval_ids = sorted(set(gt_by_id) & set(pred_by_id))
    if args.max_review_rows and args.max_review_rows > 0:
        review_ids = eval_ids[: args.max_review_rows]
    else:
        review_ids = eval_ids

    detail_rows: List[Dict[str, Any]] = []
    for rid in eval_ids:
        gt = gt_by_id[rid]
        pred = pred_by_id[rid]
        sample = sample_by_id.get(rid, {})
        gt_class = str(gt.get("coarse_class", "unknown"))
        pred_class = str(pred.get("coarse_class", "unknown"))
        gt_vis = str(gt.get("visibility", "ambiguous"))
        pred_vis = str(pred.get("visibility", "ambiguous"))
        gt_tags = normalize_tags(gt.get("visual_evidence_tags"))
        pred_tags = normalize_tags(pred.get("visual_evidence_tags"))
        short, snippet, joined_text = text_fields(pred)
        gt_short, gt_snippet, gt_text = text_fields(gt)
        row: Dict[str, Any] = {
            "record_id": rid,
            "image_id": gt.get("image_id", ""),
            "crop_path": gt.get("crop_path", ""),
            "gt_coarse_class": gt_class,
            "pred_coarse_class": pred_class,
            "coarse_class_correct": gt_class == pred_class,
            "gt_visibility": gt_vis,
            "pred_visibility": pred_vis,
            "visibility_correct": gt_vis == pred_vis,
            "gt_needs_review": gt.get("needs_review", ""),
            "pred_needs_review": pred.get("needs_review", ""),
            "needs_review_correct": parse_bool(gt.get("needs_review")) == parse_bool(pred.get("needs_review")),
            "gt_tags": ";".join(gt_tags),
            "pred_tags": ";".join(pred_tags),
            "tag_jaccard": jaccard(gt_tags, pred_tags),
            "tag_exact": gt_tags == pred_tags,
            "pred_short_description": short,
            "pred_report_snippet": snippet,
            "gt_short_description": gt_short,
            "description_present": bool(joined_text.strip()),
            "description_length": len(joined_text.strip()),
            "description_generic_auto": generic_description_flag(joined_text),
            "class_text_consistency_auto": class_text_consistency_auto(pred_class, joined_text, pred_tags),
            "parse_status": sample.get("parse_status", "from_pred_jsonl"),
            "schema_valid": sample.get("schema_valid", "from_pred_jsonl"),
        }
        detail_rows.append(row)

    # Add manual scores if present.
    if args.manual_review_csv and args.manual_review_csv.exists():
        manual = read_manual_scores(args.manual_review_csv)
        for row in detail_rows:
            m = manual.get(str(row["record_id"]), {})
            for field in MANUAL_SCORE_FIELDS + ["reviewer_notes"]:
                if field in m:
                    row[field] = m[field]

    total = len(detail_rows)
    metrics = {
        "run_name": args.run_name,
        "paths": {
            "gt_jsonl": str(args.gt_jsonl),
            "pred_jsonl": str(args.pred_jsonl),
            "sample_results_jsonl": str(args.sample_results_jsonl) if args.sample_results_jsonl else None,
            "manual_review_csv": str(args.manual_review_csv) if args.manual_review_csv else None,
        },
        "counts": {
            "gt_total": len(gt_by_id),
            "pred_total": len(pred_by_id),
            "evaluated_total": total,
            "missing_predictions": len(set(gt_by_id) - set(pred_by_id)),
            "extra_predictions_without_gt": len(set(pred_by_id) - set(gt_by_id)),
        },
        "rates": {
            "coarse_class_accuracy": safe_div(sum(bool(r["coarse_class_correct"]) for r in detail_rows), total),
            "visibility_accuracy": safe_div(sum(bool(r["visibility_correct"]) for r in detail_rows), total),
            "needs_review_accuracy": safe_div(sum(bool(r["needs_review_correct"]) for r in detail_rows), total),
            "tag_exact_rate": safe_div(sum(bool(r["tag_exact"]) for r in detail_rows), total),
            "tag_mean_jaccard": safe_div(sum(float(r["tag_jaccard"]) for r in detail_rows), total),
            "description_present_rate": safe_div(sum(bool(r["description_present"]) for r in detail_rows), total),
            "description_generic_auto_rate": safe_div(sum(bool(r["description_generic_auto"]) for r in detail_rows), total),
        },
        "manual_scores": aggregate_manual(detail_rows),
    }

    write_csv(run_dir / "structured_eval_details.csv", detail_rows)
    write_json(run_dir / "structured_eval_metrics.json", metrics)

    # Manual review template: put wrong class, generic descriptions, low tag overlap first.
    review_rows = sorted(
        detail_rows,
        key=lambda r: (bool(r["coarse_class_correct"]), -float(r["tag_jaccard"]), not bool(r["description_generic_auto"])),
    )
    review_template: List[Dict[str, Any]] = []
    for row in review_rows[: len(review_ids)]:
        item = {
            "record_id": row["record_id"],
            "image_id": row["image_id"],
            "crop_path": row["crop_path"],
            "gt_coarse_class": row["gt_coarse_class"],
            "pred_coarse_class": row["pred_coarse_class"],
            "coarse_class_correct": row["coarse_class_correct"],
            "gt_tags": row["gt_tags"],
            "pred_tags": row["pred_tags"],
            "tag_jaccard": row["tag_jaccard"],
            "pred_short_description": row["pred_short_description"],
            "pred_report_snippet": row["pred_report_snippet"],
            "class_text_consistency_auto": row["class_text_consistency_auto"],
            "description_generic_auto": row["description_generic_auto"],
        }
        for field in MANUAL_SCORE_FIELDS:
            item[field] = ""
        item["reviewer_notes"] = ""
        review_template.append(item)
    write_csv(run_dir / "manual_review_template.csv", review_template)

    summary_rows = [
        {"metric": key, "value": value}
        for key, value in metrics["rates"].items()
    ]
    manual_summary_rows = []
    for field, data in metrics["manual_scores"].items():
        manual_summary_rows.append({"manual_metric": field, "n_scored": data["n_scored"], "mean": data["mean"], "score_counts": data["score_counts"]})

    lines = [
        f"# Structured output evaluation: {args.run_name}",
        "",
        "## Automatic metrics",
        markdown_table(summary_rows, ["metric", "value"]),
        "",
        "## Manual rubric summary",
        markdown_table(manual_summary_rows, ["manual_metric", "n_scored", "mean", "score_counts"]),
        "",
        "## Artifacts",
        f"- details: `{run_dir / 'structured_eval_details.csv'}`",
        f"- metrics: `{run_dir / 'structured_eval_metrics.json'}`",
        f"- manual review template: `{run_dir / 'manual_review_template.csv'}`",
        "",
        "Manual score convention: 0 = bad, 1 = partial/uncertain, 2 = good. For hallucination, 2 means no clear hallucination.",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {run_dir / 'summary.md'}")
    print(f"Wrote: {run_dir / 'manual_review_template.csv'}")


if __name__ == "__main__":
    main()
