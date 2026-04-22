#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create visualization dashboard for Stage 3 eval artifacts."
    )
    parser.add_argument(
        "--eval-dir",
        required=True,
        help="Path to eval directory containing metrics.json/review_table.csv/confusion_*.csv",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for plots/report (default: <eval-dir>/visuals)",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_confusion_csv(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError(f"Confusion CSV has no data rows: {path}")
    col_labels = rows[0][1:]
    row_labels = []
    values: list[list[float]] = []
    for row in rows[1:]:
        row_labels.append(row[0])
        values.append([float(x) for x in row[1:]])
    return row_labels, col_labels, np.array(values, dtype=float)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    out_path: Path,
    cmap: str = "Blues",
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_title(title, pad=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            color = "white" if v > (matrix.max() * 0.6 if matrix.max() > 0 else 0) else "black"
            ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=9, color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def save_kpi_bar(metrics: dict, out_path: Path) -> None:
    rates = metrics.get("rates", {})
    f1 = metrics.get("f1", {})

    labels = [
        "Parse success",
        "Schema valid",
        "Coarse acc",
        "Coarse macro-F1",
        "Visibility acc",
        "Visibility macro-F1",
        "Needs-review acc",
        "Tag mean Jaccard",
    ]
    values = [
        float(rates.get("parse_success_rate", 0.0)),
        float(rates.get("schema_valid_rate", 0.0)),
        float(rates.get("coarse_class_accuracy", 0.0)),
        float(f1.get("coarse_class_macro_f1", 0.0)),
        float(rates.get("visibility_accuracy", 0.0)),
        float(f1.get("visibility_macro_f1", 0.0)),
        float(rates.get("needs_review_accuracy", 0.0)),
        float(rates.get("tag_mean_jaccard", 0.0)),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color="#1f77b4")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Score")
    ax.set_title("Stage 3 KPI Overview", pad=12)
    ax.grid(axis="x", alpha=0.25)

    for b, v in zip(bars, values):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def save_visibility_bias_chart(metrics: dict, review_df: pd.DataFrame, out_path: Path) -> None:
    gt = review_df["gt_visibility"].value_counts().to_dict()
    pred = review_df["pred_visibility"].value_counts().to_dict()
    labels = ["clear", "partial", "ambiguous"]
    gt_vals = [int(gt.get(x, 0)) for x in labels]
    pred_vals = [int(pred.get(x, 0)) for x in labels]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(x - width / 2, gt_vals, width, label="GT", color="#2ca02c")
    ax.bar(x + width / 2, pred_vals, width, label="Pred", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count")
    ax.set_title("Visibility Distribution: GT vs Pred", pad=10)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    pred_amb = float(metrics.get("rates", {}).get("pred_ambiguous_rate", 0.0))
    gt_amb = float(metrics.get("rates", {}).get("gt_ambiguous_rate", 0.0))
    ax.text(
        0.02,
        0.95,
        f"ambiguous rate: pred={pred_amb:.3f}, gt={gt_amb:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.7, "edgecolor": "#aaaaaa"},
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def save_failure_modes(review_df: pd.DataFrame, out_path: Path) -> dict[str, int]:
    df = review_df.copy()
    for c in ["coarse_class_match", "visibility_match", "needs_review_match"]:
        df[c] = df[c].astype(str).str.lower() == "true"

    modes = {
        "all_match": int((df["coarse_class_match"] & df["visibility_match"] & df["needs_review_match"]).sum()),
        "coarse_only_mismatch": int((~df["coarse_class_match"] & df["visibility_match"] & df["needs_review_match"]).sum()),
        "visibility_only_mismatch": int((df["coarse_class_match"] & ~df["visibility_match"] & df["needs_review_match"]).sum()),
        "needs_review_only_mismatch": int((df["coarse_class_match"] & df["visibility_match"] & ~df["needs_review_match"]).sum()),
        "multiple_mismatch": int((~(df["coarse_class_match"] & df["visibility_match"] & df["needs_review_match"])).sum())
        - int((~df["coarse_class_match"] & df["visibility_match"] & df["needs_review_match"]).sum())
        - int((df["coarse_class_match"] & ~df["visibility_match"] & df["needs_review_match"]).sum())
        - int((df["coarse_class_match"] & df["visibility_match"] & ~df["needs_review_match"]).sum()),
    }

    labels = list(modes.keys())
    vals = [modes[k] for k in labels]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(labels, vals, color=["#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"])
    ax.set_title("Failure Modes", pad=10)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, str(v), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)

    return modes


def decide_prompt_action(metrics: dict) -> tuple[str, list[str]]:
    rates = metrics.get("rates", {})
    f1 = metrics.get("f1", {})
    reasons: list[str] = []

    coarse_acc = float(rates.get("coarse_class_accuracy", 0.0))
    visibility_macro_f1 = float(f1.get("visibility_macro_f1", 0.0))
    pred_amb = float(rates.get("pred_ambiguous_rate", 0.0))
    gt_amb = float(rates.get("gt_ambiguous_rate", 0.0))
    tag_j = float(rates.get("tag_mean_jaccard", 0.0))

    if visibility_macro_f1 < 0.6:
        reasons.append(f"visibility macro-F1 is low ({visibility_macro_f1:.3f})")
    if gt_amb > 0 and pred_amb < gt_amb * 0.6:
        reasons.append(f"model underpredicts ambiguous visibility (pred={pred_amb:.3f}, gt={gt_amb:.3f})")
    if coarse_acc < 0.9:
        reasons.append(f"coarse-class accuracy is below 0.90 ({coarse_acc:.3f})")
    if tag_j < 0.3:
        reasons.append(f"tag overlap is weak (mean Jaccard={tag_j:.3f})")

    if reasons:
        return "IMPROVE_PROMPT", reasons
    return "PROMPT_OK_MOVE_TO_NEXT_STAGE", ["current metrics are stable enough for next stage"]


def write_markdown_report(
    out_path: Path,
    metrics: dict,
    verdict: str,
    reasons: list[str],
    modes: dict[str, int],
) -> None:
    lines = []
    lines.append("# Stage 3 Visual Report")
    lines.append("")
    lines.append("## Verdict")
    lines.append(f"- `{verdict}`")
    for r in reasons:
        lines.append(f"- {r}")
    lines.append("")
    lines.append("## Key Scores")
    rates = metrics.get("rates", {})
    f1 = metrics.get("f1", {})
    lines.append(f"- Coarse accuracy: `{float(rates.get('coarse_class_accuracy', 0.0)):.3f}`")
    lines.append(f"- Coarse macro-F1: `{float(f1.get('coarse_class_macro_f1', 0.0)):.3f}`")
    lines.append(f"- Visibility accuracy: `{float(rates.get('visibility_accuracy', 0.0)):.3f}`")
    lines.append(f"- Visibility macro-F1: `{float(f1.get('visibility_macro_f1', 0.0)):.3f}`")
    lines.append(f"- Needs-review accuracy: `{float(rates.get('needs_review_accuracy', 0.0)):.3f}`")
    lines.append(f"- Tag mean Jaccard: `{float(rates.get('tag_mean_jaccard', 0.0)):.3f}`")
    lines.append("")
    lines.append("## Failure Modes")
    for k, v in modes.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Plots")
    lines.append("![KPI](kpi_overview.png)")
    lines.append("")
    lines.append("![Coarse confusion](confusion_coarse_class.png)")
    lines.append("")
    lines.append("![Visibility confusion](confusion_visibility.png)")
    lines.append("")
    lines.append("![Visibility bias](visibility_bias.png)")
    lines.append("")
    lines.append("![Failure modes](failure_modes.png)")
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir).resolve()
    if not eval_dir.exists():
        raise FileNotFoundError(f"Eval dir not found: {eval_dir}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else eval_dir / "visuals"
    ensure_dir(out_dir)

    metrics_path = eval_dir / "metrics.json"
    review_table_path = eval_dir / "review_table.csv"
    conf_coarse_path = eval_dir / "confusion_coarse_class.csv"
    conf_visibility_path = eval_dir / "confusion_visibility.csv"

    metrics = read_json(metrics_path)
    review_df = pd.read_csv(review_table_path)
    coarse_rows, coarse_cols, coarse_mat = load_confusion_csv(conf_coarse_path)
    vis_rows, vis_cols, vis_mat = load_confusion_csv(conf_visibility_path)

    save_kpi_bar(metrics, out_dir / "kpi_overview.png")
    save_heatmap(
        coarse_mat,
        coarse_rows,
        coarse_cols,
        "Coarse Class Confusion",
        out_dir / "confusion_coarse_class.png",
    )
    save_heatmap(
        vis_mat,
        vis_rows,
        vis_cols,
        "Visibility Confusion",
        out_dir / "confusion_visibility.png",
    )
    save_visibility_bias_chart(metrics, review_df, out_dir / "visibility_bias.png")
    failure_modes = save_failure_modes(review_df, out_dir / "failure_modes.png")

    verdict, reasons = decide_prompt_action(metrics)
    verdict_payload = {"verdict": verdict, "reasons": reasons}
    (out_dir / "verdict.json").write_text(
        json.dumps(verdict_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    write_markdown_report(
        out_dir / "report.md",
        metrics=metrics,
        verdict=verdict,
        reasons=reasons,
        modes=failure_modes,
    )

    print(f"Visual report generated: {out_dir}")
    print(f"Verdict: {verdict}")
    for reason in reasons:
        print(f"- {reason}")


if __name__ == "__main__":
    main()

