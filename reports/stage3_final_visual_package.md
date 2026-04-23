# Stage 3 Final Visual Package

This repository now includes an extended Stage 3 visualization pass that generates a consolidated report with charts and tables from eval artifacts.

Stage 3 freeze context:

- frozen prompt: `qwen_vlm_labels_v1_prompt_v6d_balanced_notaglock`
- reference run: `outputs/stage3_vlm_baseline_runs/stage3_qwen_val_v2_kaggle`
- prompt decision checkpoints:
  - `reports/stage3_prompt_sweep_v6_checkpoint.md`
  - `reports/stage3_v6d_vs_v6f_checkpoint.md`

Primary script:

- `scripts/visualize_stage3_eval_results.py`

Baseline command:

```bash
python scripts/visualize_stage3_eval_results.py \
  --eval-dir outputs/stage3_vlm_baseline_runs/<run_id>/eval
```

With sweep + ablation context:

```bash
python scripts/visualize_stage3_eval_results.py \
  --eval-dir outputs/stage3_vlm_baseline_runs/<run_id>/eval \
  --sweep-csv reports/stage3_prompt_sweep_v6_checkpoint/prompt_sweep_comparison.csv \
  --ablation-csv reports/stage3_v6d_vs_v6f_checkpoint/v6d_vs_v6f_comparison.csv
```

What is generated (`<run_id>/eval/visuals`):

- confusion heatmaps (count and row-normalized)
- KPI and per-label F1 charts
- GT-vs-pred distributions (coarse class, visibility)
- mismatch-rate charts by GT coarse class and GT visibility
- failure mode chart + failure mode table
- text length diagnostics
- tag diagnostics (frequency and Jaccard by class)
- optional sweep and ablation comparison charts
- final report: `report.md`
- machine-readable tables: `table_*.csv`
