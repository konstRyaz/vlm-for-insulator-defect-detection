#!/usr/bin/env python3
"""Run no-VLM feature-extractor baselines for insulator crop classification.

This script is intentionally parallel to the existing Stage 3/Stage 4 notebooks:
- load the clean crop-level train/val JSONL;
- resolve crop_path without exposing class labels to models;
- extract frozen visual features from HF or timm backbones;
- select classifier hyperparameters by train CV only;
- evaluate once on val;
- write leaderboard, predictions, confusion matrices and a markdown summary.

The goal is to compare direct/hybrid VLM systems against classical no-VLM CV baselines.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

LABELS_DEFAULT = ["insulator_ok", "defect_flashover", "defect_broken"]


@dataclass
class ModelSpec:
    key: str
    kind: str  # hf_auto or timm
    model_id: str
    batch_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run non-VLM crop classifier baselines.")
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--val-jsonl", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/next_research/non_vlm_baselines"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/next_research/feature_cache"))
    parser.add_argument("--labels", default=",".join(LABELS_DEFAULT), help="Comma-separated class list.")
    parser.add_argument(
        "--hf-models",
        default="dinov2_base=facebook/dinov2-base:16,clip_b32=openai/clip-vit-base-patch32:32,clip_l14=openai/clip-vit-large-patch14:16,siglip_b16=google/siglip-base-patch16-224:16",
        help="Comma-separated key=model_id:batch entries. Use empty string to disable.",
    )
    parser.add_argument(
        "--timm-models",
        default="resnet50=resnet50.a1_in1k:32,efficientnet_b0=efficientnet_b0.ra_in1k:32,convnext_tiny=convnext_tiny.fb_in1k:16",
        help="Comma-separated key=model_id:batch entries. Use empty string to disable.",
    )
    parser.add_argument("--classifiers", default="logreg,svm", help="Comma-separated: logreg,svm,mlp")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-recompute-features", action="store_true")
    parser.add_argument("--max-train", type=int, default=0, help="Debug limit. 0 means all.")
    parser.add_argument("--max-val", type=int, default=0, help="Debug limit. 0 means all.")
    parser.add_argument("--skip-timm-on-import-error", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
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
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(obj)
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_model_specs(raw: str, kind: str) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    if not raw.strip():
        return specs
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Model spec must be key=model_id:batch, got: {item}")
        key, rest = item.split("=", 1)
        if ":" in rest:
            model_id, batch_s = rest.rsplit(":", 1)
            batch_size = int(batch_s)
        else:
            model_id, batch_size = rest, 16
        specs.append(ModelSpec(key=key.strip(), kind=kind, model_id=model_id.strip(), batch_size=batch_size))
    return specs


def resolve_crop_path(row: Dict[str, Any], jsonl_path: Path, dataset_root: Optional[Path]) -> Path:
    raw = row.get("crop_path") or row.get("image_path")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Row has no crop_path/image_path: {row.get('record_id')}")
    p = Path(raw)
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    candidates.extend([
        jsonl_path.parent / p,
        jsonl_path.parent.parent / p,
        Path.cwd() / p,
    ])
    if dataset_root is not None:
        candidates.extend([dataset_root / p, dataset_root / p.name])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve crop path for {row.get('record_id')}: {raw}")


def filter_rows(rows: List[Dict[str, Any]], labels: Sequence[str], limit: int) -> List[Dict[str, Any]]:
    out = [r for r in rows if r.get("coarse_class") in labels]
    if limit and limit > 0:
        out = out[:limit]
    return out


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return safe_div(sum(1 for a, b in zip(y_true, y_pred) if a == b), len(y_true))


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> Tuple[float, Dict[str, float]]:
    per: Dict[str, float] = {}
    vals: List[float] = []
    for label in labels:
        tp = sum(1 for g, p in zip(y_true, y_pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(y_true, y_pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(y_true, y_pred) if g == label and p != label)
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom else 0.0
        per[label] = f1
        vals.append(f1)
    return safe_div(sum(vals), len(vals)), per


def recall_by_class(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> Dict[str, float]:
    return {
        label: safe_div(
            sum(1 for g, p in zip(y_true, y_pred) if g == label and p == label),
            sum(1 for g in y_true if g == label),
        )
        for label in labels
    }


def confusion_matrix_rows(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> List[Dict[str, Any]]:
    pred_labels = list(labels) + ["unknown", "other", "__missing__"]
    rows: List[Dict[str, Any]] = []
    for gt_label in labels:
        row: Dict[str, Any] = {"gt\\pred": gt_label}
        for pred_label in pred_labels:
            row[pred_label] = 0
        rows.append(row)
    by_gt = {r["gt\\pred"]: r for r in rows}
    for gt, pred in zip(y_true, y_pred):
        if gt not in by_gt:
            continue
        pp = pred if pred in pred_labels else "other"
        by_gt[gt][pp] += 1
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def npz_path(cache_dir: Path, split: str, spec: ModelSpec) -> Path:
    safe_key = spec.key.replace("/", "_").replace(":", "_")
    return cache_dir / safe_key / f"{split}_features.npz"


def load_hf_features(spec: ModelSpec, image_paths: Sequence[Path], cache_path: Path, force: bool) -> np.ndarray:
    if cache_path.exists() and not force:
        return np.load(cache_path)["features"]
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel, AutoProcessor

    def load_processor(model_id: str):
        try:
            return AutoProcessor.from_pretrained(model_id)
        except Exception:
            return AutoImageProcessor.from_pretrained(model_id)

    def image_features(model: Any, inputs: Any) -> Any:
        if hasattr(model, "get_image_features"):
            return model.get_image_features(**inputs)
        out = model(**inputs)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state.mean(dim=1)
        raise RuntimeError(f"Cannot infer feature tensor for {spec.model_id}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = load_processor(spec.model_id)
    model = AutoModel.from_pretrained(spec.model_id).to(device)
    model.eval()
    feats: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(image_paths), spec.batch_size):
            batch_paths = image_paths[start : start + spec.batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
            out = image_features(model, inputs).float()
            out = torch.nn.functional.normalize(out, dim=-1)
            feats.append(out.detach().cpu().numpy())
            print(f"{spec.key}: embedded {min(start + spec.batch_size, len(image_paths))}/{len(image_paths)}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    arr = np.concatenate(feats, axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=arr)
    return arr


def load_timm_features(spec: ModelSpec, image_paths: Sequence[Path], cache_path: Path, force: bool) -> np.ndarray:
    if cache_path.exists() and not force:
        return np.load(cache_path)["features"]
    import torch
    from PIL import Image
    import timm
    from timm.data import create_transform, resolve_model_data_config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(spec.model_id, pretrained=True, num_classes=0).to(device)
    model.eval()
    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)
    feats: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(image_paths), spec.batch_size):
            batch_paths = image_paths[start : start + spec.batch_size]
            images = [transform(Image.open(p).convert("RGB")) for p in batch_paths]
            x = torch.stack(images).to(device)
            out = model(x).float()
            out = torch.nn.functional.normalize(out, dim=-1)
            feats.append(out.detach().cpu().numpy())
            print(f"{spec.key}: embedded {min(start + spec.batch_size, len(image_paths))}/{len(image_paths)}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    arr = np.concatenate(feats, axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=arr)
    return arr


def load_features(spec: ModelSpec, split: str, image_paths: Sequence[Path], args: argparse.Namespace) -> np.ndarray:
    cache_path = npz_path(args.cache_dir, split, spec)
    if spec.kind == "hf_auto":
        return load_hf_features(spec, image_paths, cache_path, args.force_recompute_features)
    if spec.kind == "timm":
        return load_timm_features(spec, image_paths, cache_path, args.force_recompute_features)
    raise ValueError(f"Unknown model kind: {spec.kind}")


def min_class_count(y: Sequence[str]) -> int:
    counts = Counter(y)
    return min(counts.values()) if counts else 0


def classifier_candidates(kind: str, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if kind == "logreg":
        for C in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
            for class_weight in [None, "balanced"]:
                clf = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=C, class_weight=class_weight, max_iter=5000, random_state=seed),
                )
                yield {"classifier": "logreg", "C": C, "class_weight": class_weight or "none"}, clf
    elif kind == "svm":
        for C in [0.1, 0.3, 1.0, 3.0]:
            for kernel in ["linear", "rbf"]:
                for class_weight in [None, "balanced"]:
                    clf = make_pipeline(
                        StandardScaler(),
                        SVC(C=C, kernel=kernel, class_weight=class_weight, probability=True, random_state=seed),
                    )
                    yield {"classifier": "svm", "C": C, "kernel": kernel, "class_weight": class_weight or "none"}, clf
    elif kind == "mlp":
        for hidden in [(64,), (128,)]:
            for alpha in [0.0001, 0.001, 0.01]:
                clf = make_pipeline(
                    StandardScaler(),
                    MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, max_iter=1000, early_stopping=True, random_state=seed),
                )
                yield {"classifier": "mlp", "hidden": "x".join(map(str, hidden)), "alpha": alpha}, clf
    else:
        raise ValueError(f"Unknown classifier kind: {kind}")


def cv_select_classifier(X: np.ndarray, y: np.ndarray, labels: Sequence[str], classifier_kinds: Sequence[str], n_splits: int, seed: int):
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold

    actual_splits = min(n_splits, min_class_count(y))
    if actual_splits < 2:
        raise ValueError(f"Not enough samples per class for CV: min_class_count={min_class_count(y)}")
    splitter = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=seed)
    rows: List[Dict[str, Any]] = []
    best_score_tuple: Tuple[float, float, float, float] = (-1, -1, -1, -1)
    best_meta: Optional[Dict[str, Any]] = None
    best_clf_template: Any = None

    for kind in classifier_kinds:
        for meta, clf_template in classifier_candidates(kind, seed):
            fold_metrics: List[Dict[str, float]] = []
            for train_idx, cv_idx in splitter.split(X, y):
                clf = clone(clf_template)
                clf.fit(X[train_idx], y[train_idx])
                pred = clf.predict(X[cv_idx])
                m = compute_metrics(y[cv_idx].tolist(), pred.tolist(), labels)
                fold_metrics.append(m)
            row: Dict[str, Any] = dict(meta)
            for key in ["accuracy", "macro_f1", "recall_insulator_ok", "recall_defect_flashover", "recall_defect_broken"]:
                vals = [m[key] for m in fold_metrics]
                row[f"cv_{key}_mean"] = float(np.mean(vals))
                row[f"cv_{key}_std"] = float(np.std(vals))
            rows.append(row)
            score_tuple = (
                row["cv_macro_f1_mean"],
                row["cv_accuracy_mean"],
                row.get("cv_recall_defect_flashover_mean", 0.0),
                row.get("cv_recall_defect_broken_mean", 0.0),
            )
            if score_tuple > best_score_tuple:
                best_score_tuple = score_tuple
                best_meta = row
                best_clf_template = clf_template
    if best_meta is None or best_clf_template is None:
        raise RuntimeError("No classifier candidates produced results")
    return best_meta, best_clf_template, rows


def compute_metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> Dict[str, float]:
    macro, _ = macro_f1(y_true, y_pred, labels)
    recalls = recall_by_class(y_true, y_pred, labels)
    out = {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro,
    }
    for label in labels:
        out[f"recall_{label}"] = recalls[label]
    # Stable aliases for the three target classes.
    out["recall_insulator_ok"] = recalls.get("insulator_ok", 0.0)
    out["recall_defect_flashover"] = recalls.get("defect_flashover", 0.0)
    out["recall_defect_broken"] = recalls.get("defect_broken", 0.0)
    return out


def predict_proba_or_empty(clf: Any, X: np.ndarray) -> Tuple[List[str], Optional[np.ndarray], List[str]]:
    pred = clf.predict(X).tolist()
    if hasattr(clf, "predict_proba"):
        try:
            probs = clf.predict_proba(X)
            classes = list(getattr(clf, "classes_", []))
            # Pipeline exposes classes_ after last step in recent sklearn; fallback below.
            if not classes and hasattr(clf, "steps"):
                classes = list(getattr(clf.steps[-1][1], "classes_", []))
            return pred, probs, classes
        except Exception:
            return pred, None, []
    return pred, None, []


def run_one_model(spec: ModelSpec, args: argparse.Namespace, labels: Sequence[str], train_paths: Sequence[Path], val_paths: Sequence[Path], y_train: np.ndarray, y_val: np.ndarray, val_ids: Sequence[str]) -> Dict[str, Any]:
    from sklearn.base import clone

    out_dir = args.out_dir / spec.key
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== {spec.key} ({spec.kind}) {spec.model_id} ===")
    X_train = load_features(spec, "train", train_paths, args)
    X_val = load_features(spec, "val", val_paths, args)
    classifier_kinds = [x.strip() for x in args.classifiers.split(",") if x.strip()]
    best_meta, best_template, cv_rows = cv_select_classifier(X_train, y_train, labels, classifier_kinds, args.cv_splits, args.seed)
    write_csv(out_dir / "cv_results.csv", cv_rows)
    clf = clone(best_template)
    clf.fit(X_train, y_train)
    pred, probs, prob_classes = predict_proba_or_empty(clf, X_val)
    val_metrics = compute_metrics(y_val.tolist(), pred, labels)
    cm_rows = confusion_matrix_rows(y_val.tolist(), pred, labels)
    write_csv(out_dir / "confusion_matrix.csv", cm_rows)

    pred_rows: List[Dict[str, Any]] = []
    for i, (rid, gt, pr) in enumerate(zip(val_ids, y_val.tolist(), pred)):
        row: Dict[str, Any] = {
            "record_id": rid,
            "gt": gt,
            "pred_coarse_class": pr,
            "correct": str(gt == pr).lower(),
        }
        if probs is not None:
            row["confidence"] = float(np.max(probs[i]))
            for cls, val in zip(prob_classes, probs[i]):
                row[f"prob_{cls}"] = float(val)
        pred_rows.append(row)
    write_csv(out_dir / "val_predictions.csv", pred_rows)

    result: Dict[str, Any] = {
        "run_id": f"non_vlm_{spec.key}_{best_meta['classifier']}",
        "model_key": spec.key,
        "kind": spec.kind,
        "model_id": spec.model_id,
        "status": "ok",
        **best_meta,
        **val_metrics,
        "cv_results_path": str(out_dir / "cv_results.csv"),
        "predictions_path": str(out_dir / "val_predictions.csv"),
        "confusion_matrix_path": str(out_dir / "confusion_matrix.csv"),
    }
    write_json(out_dir / "result_summary.json", result)
    return result


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
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    train_rows = filter_rows(read_jsonl(args.train_jsonl), labels, args.max_train)
    val_rows = filter_rows(read_jsonl(args.val_jsonl), labels, args.max_val)
    if not train_rows or not val_rows:
        raise ValueError("Train/val rows are empty after filtering")

    train_paths = [resolve_crop_path(r, args.train_jsonl, args.dataset_root) for r in train_rows]
    val_paths = [resolve_crop_path(r, args.val_jsonl, args.dataset_root) for r in val_rows]
    y_train = np.array([str(r["coarse_class"]) for r in train_rows])
    y_val = np.array([str(r["coarse_class"]) for r in val_rows])
    val_ids = [str(r.get("record_id")) for r in val_rows]

    distribution_rows = []
    for split_name, y in [("train", y_train), ("val", y_val)]:
        counts = Counter(y.tolist())
        for label in labels:
            distribution_rows.append({"split": split_name, "label": label, "count": counts.get(label, 0)})
    write_csv(args.out_dir / "class_distribution.csv", distribution_rows)

    specs = parse_model_specs(args.hf_models, "hf_auto")
    timm_specs = parse_model_specs(args.timm_models, "timm")
    specs.extend(timm_specs)

    results: List[Dict[str, Any]] = []
    for spec in specs:
        try:
            if spec.kind == "timm" and args.skip_timm_on_import_error:
                try:
                    import timm  # noqa: F401
                except Exception as exc:
                    results.append({"model_key": spec.key, "kind": spec.kind, "model_id": spec.model_id, "status": "skipped_timm_import_error", "error": str(exc)})
                    continue
            result = run_one_model(spec, args, labels, train_paths, val_paths, y_train, y_val, val_ids)
            results.append(result)
        except Exception as exc:
            err_dir = args.out_dir / spec.key
            err_dir.mkdir(parents=True, exist_ok=True)
            (err_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            row = {"model_key": spec.key, "kind": spec.kind, "model_id": spec.model_id, "status": "error", "error": str(exc)}
            results.append(row)
            print(traceback.format_exc())
            if not args.continue_on_error:
                raise

    ok = [r for r in results if r.get("status") == "ok"]
    ok_sorted = sorted(ok, key=lambda r: (r.get("macro_f1", 0.0), r.get("accuracy", 0.0), r.get("recall_defect_flashover", 0.0), r.get("recall_defect_broken", 0.0)), reverse=True)
    other = [r for r in results if r.get("status") != "ok"]
    leaderboard = ok_sorted + other
    write_csv(args.out_dir / "leaderboard_non_vlm.csv", leaderboard)
    write_json(args.out_dir / "run_manifest.json", {"labels": labels, "train_jsonl": str(args.train_jsonl), "val_jsonl": str(args.val_jsonl), "results": results})

    fields = ["run_id", "model_key", "kind", "model_id", "classifier", "C", "kernel", "class_weight", "accuracy", "macro_f1", "recall_insulator_ok", "recall_defect_flashover", "recall_defect_broken", "status"]
    lines = [
        "# Non-VLM baseline sweep",
        "",
        "This report compares frozen visual feature extractors plus classical classifiers against VLM/hybrid systems.",
        "Hyperparameters were selected by train CV only. Validation was evaluated once per selected configuration.",
        "",
        "## Class distribution",
        markdown_table(distribution_rows, ["split", "label", "count"]),
        "",
        "## Leaderboard",
        markdown_table(leaderboard, fields),
        "",
        "## Interpretation placeholder",
        "Compare this table against Qwen direct and DINOv2+Qwen hybrid in `reports/next_research/vlm_benefit/`.",
    ]
    (args.out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {args.out_dir / 'leaderboard_non_vlm.csv'}")
    print(f"Wrote: {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
