#!/usr/bin/env python3
"""Binary ablation for the main error boundary: insulator_ok vs defect_flashover.

This script reuses the same idea as no-VLM baselines, but filters to two classes
and evaluates whether a dedicated binary classifier can reduce the main confusion.
Use results as diagnostic unless thresholds are selected by train CV only.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run binary normal-vs-flashover ablation via run_non_vlm_baselines.py.")
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--val-jsonl", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/next_research/accuracy_ablation/flashover_binary"))
    parser.add_argument("--script", type=Path, default=Path("scripts/run_non_vlm_baselines.py"))
    parser.add_argument("--models", default="dinov2_base=facebook/dinov2-base:16,clip_b32=openai/clip-vit-base-patch32:32")
    parser.add_argument("--classifiers", default="logreg,svm")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        sys.executable,
        str(args.script),
        "--train-jsonl", str(args.train_jsonl),
        "--val-jsonl", str(args.val_jsonl),
        "--out-dir", str(args.out_dir),
        "--labels", "insulator_ok,defect_flashover",
        "--hf-models", args.models,
        "--timm-models", "",
        "--classifiers", args.classifiers,
        "--continue-on-error",
    ]
    if args.dataset_root:
        cmd.extend(["--dataset-root", str(args.dataset_root)])
    print("$", " ".join(cmd))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "binary_ablation_run.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, text=True, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        msg = f"run_non_vlm_baselines failed with code {proc.returncode}. See {log_path}"
        if args.continue_on_error:
            print(msg)
            return
        raise SystemExit(msg)
    note = [
        "# Flashover binary ablation",
        "",
        "This diagnostic run filters labels to `insulator_ok` and `defect_flashover`, the main remaining error boundary.",
        "Use it to decide whether a dedicated review/policy layer is worth testing in the full 3-class pipeline.",
        "",
        f"Run log: `{log_path}`",
        f"Leaderboard: `{args.out_dir / 'leaderboard_non_vlm.csv'}`",
    ]
    (args.out_dir / "README.md").write_text("\n".join(note), encoding="utf-8")
    print(f"Wrote: {args.out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
