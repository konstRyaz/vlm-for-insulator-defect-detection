#!/usr/bin/env python3
"""Build a comparative report: no-VLM vs direct VLM vs hybrid.

The script is deliberately tolerant to partial inputs. Codex can call it after
running non-VLM baselines and structured JSON evaluation. It produces a report
that frames the VLM benefit around both classification and structured output.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build VLM benefit report.")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/next_research/vlm_benefit"))
    parser.add_argument("--non-vlm-leaderboard", type=Path, default=Path("reports/next_research/non_vlm_baselines/leaderboard_non_vlm.csv"))
    parser.add_argument("--current-registry", type=Path, default=Path("reports/operation_next/experiment_registry.csv"))
    parser.add_argument("--structured-eval-root", type=Path, default=Path("reports/next_research/structured_output_eval"))
    parser.add_argument("--manual-benefit-csv", type=Path, default=None, help="Optional manually curated system table.")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_float(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return value
    s = str(value).strip()
    if not s:
        return ""
    try:
        return float(s)
    except ValueError:
        return s


def best_non_vlm(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        return {}
    def key(row: Dict[str, str]):
        return (
            float(row.get("macro_f1") or 0),
            float(row.get("accuracy") or 0),
            float(row.get("recall_defect_flashover") or 0),
            float(row.get("recall_defect_broken") or 0),
        )
    return max(ok, key=key)


def registry_systems(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    wanted = []
    for r in rows:
        rid = r.get("run_id", "")
        if rid in {
            "stage4_context_pad030_maxpix401k",
            "stage4_dinov2_packfix_secondbest035",
            "stage3_qwen_val_v2_clean_final",
        } or "qwen25vl_3b_control" in rid or "internvl3_2b_base" in rid:
            wanted.append(r)
    return wanted


def collect_structured_metrics(root: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not root.exists():
        return out
    for p in sorted(root.glob("*/structured_eval_metrics.json")):
        data = read_json(p)
        rates = data.get("rates", {}) if isinstance(data, dict) else {}
        manual = data.get("manual_scores", {}) if isinstance(data, dict) else {}
        row: Dict[str, Any] = {"run_name": data.get("run_name", p.parent.name)}
        for key in ["coarse_class_accuracy", "visibility_accuracy", "tag_mean_jaccard", "description_present_rate", "description_generic_auto_rate"]:
            row[key] = rates.get(key, "")
        for key in ["manual_description_relevance", "manual_visual_evidence_score", "manual_hallucination_score", "manual_usefulness_score", "manual_class_text_consistency"]:
            item = manual.get(key, {}) if isinstance(manual, dict) else {}
            row[f"{key}_mean"] = item.get("mean", "") if isinstance(item, dict) else ""
            row[f"{key}_n"] = item.get("n_scored", "") if isinstance(item, dict) else ""
        out.append(row)
    return out


def markdown_table(rows: List[Dict[str, Any]], fields: Sequence[str]) -> str:
    if not rows:
        return "_No rows found._"
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        vals: List[str] = []
        for f in fields:
            v = row.get(f, "")
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v).replace("\n", " ")[:220])
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    non_vlm_rows = read_csv(args.non_vlm_leaderboard)
    registry_rows = read_csv(args.current_registry)
    structured_rows = collect_structured_metrics(args.structured_eval_root)

    benefit_rows: List[Dict[str, Any]] = []
    best_nv = best_non_vlm(non_vlm_rows)
    if best_nv:
        benefit_rows.append({
            "system": "best_non_vlm_classifier",
            "uses_vlm": "no",
            "uses_train": "yes_classifier_train_cv",
            "output_type": "class_only",
            "accuracy": maybe_float(best_nv.get("accuracy")),
            "macro_f1": maybe_float(best_nv.get("macro_f1")),
            "json_output": "no",
            "description_output": "no",
            "recommended_role": "class-only CV baseline",
            "source_run": best_nv.get("run_id") or best_nv.get("model_key"),
        })

    for r in registry_systems(registry_rows):
        rid = r.get("run_id", "")
        model_blob = " ".join([rid, r.get("model_family", ""), r.get("model_id", ""), r.get("notes", "")]).lower()
        if "dinov2" in model_blob and "qwen" in model_blob and "stage4" in rid:
            role = "hybrid classifier+reporter"
            uses_vlm = "partial_qwen_reporter"
            uses_train = "yes_classifier_branch"
            output_type = "JSON+class"
        elif "qwen" in model_blob or "internvl" in model_blob or "llava" in model_blob or "smolvlm" in model_blob:
            role = "direct/frozen VLM structured reporter"
            uses_vlm = "yes"
            uses_train = "no_zero_shot"
            output_type = "JSON+class"
        else:
            role = "baseline"
            uses_vlm = "unknown"
            uses_train = r.get("train_data", "")
            output_type = "unknown"
        benefit_rows.append({
            "system": rid,
            "uses_vlm": uses_vlm,
            "uses_train": uses_train,
            "output_type": output_type,
            "accuracy": maybe_float(str(r.get("accuracy", "")).replace("_pipeline_correct", "")),
            "macro_f1": r.get("macro_f1", ""),
            "json_output": "yes" if "qwen" in rid or "internvl" in rid or "dinov2" in rid else "unknown",
            "description_output": "yes_if_qwen_reporter_fields_available",
            "recommended_role": role,
            "source_run": rid,
        })

    if args.manual_benefit_csv and args.manual_benefit_csv.exists():
        benefit_rows.extend(read_csv(args.manual_benefit_csv))

    write_csv(args.out_dir / "benefit_matrix.csv", benefit_rows)
    write_csv(args.out_dir / "structured_metrics_summary.csv", structured_rows)

    classification_fields = ["system", "uses_vlm", "uses_train", "output_type", "accuracy", "macro_f1", "recommended_role"]
    structured_fields = ["run_name", "coarse_class_accuracy", "visibility_accuracy", "tag_mean_jaccard", "description_present_rate", "description_generic_auto_rate", "manual_description_relevance_mean", "manual_hallucination_score_mean", "manual_usefulness_score_mean"]

    lines = [
        "# VLM vs non-VLM benefit report",
        "",
        "## Main question",
        "",
        "Does the VLM module improve the insulator-defect pipeline compared with ordinary no-VLM computer-vision baselines?",
        "",
        "The answer must be decomposed: class accuracy and structured-reporting value are different axes.",
        "",
        "## Classification comparison",
        "",
        markdown_table(benefit_rows, classification_fields),
        "",
        "## Structured-output comparison",
        "",
        markdown_table(structured_rows, structured_fields),
        "",
        "## Interpretation template",
        "",
        "1. If no-VLM classifier is best by class accuracy, report it honestly.",
        "2. The VLM benefit is not necessarily raw accuracy; it is JSON/reporting/explanation output.",
        "3. The most defensible architecture is hybrid: visual classifier for `coarse_class`, VLM reporter for structured fields.",
        "4. If Qwen text/tags disagree with the DINOv2-overridden class, add a consistency/review flag or regenerate reporter fields conditioned on class.",
        "",
        "## Recommended claim",
        "",
        "> Direct frozen VLM is not the strongest crop-level classifier on this small specialized dataset. However, VLM remains useful as a structured reporter. The hybrid system combines the class accuracy of discriminative visual features with the reportability of VLM output.",
    ]
    (args.out_dir / "vlm_vs_non_vlm_benefit_report.md").write_text("\n".join(lines), encoding="utf-8")

    recommendation = [
        "# Final recommendation",
        "",
        "Use the next report to present three systems:",
        "",
        "1. no-VLM classifier as the class-only baseline;",
        "2. direct VLM as the zero-shot structured baseline;",
        "3. DINOv2 + Qwen as the practical hybrid architecture.",
        "",
        "Do not claim that VLM wins by raw accuracy unless the table proves it. Claim that VLM adds structured output and interpretability, while the discriminative branch handles coarse class more robustly.",
    ]
    (args.out_dir / "final_recommendation.md").write_text("\n".join(recommendation), encoding="utf-8")
    print(f"Wrote: {args.out_dir / 'vlm_vs_non_vlm_benefit_report.md'}")


if __name__ == "__main__":
    main()
