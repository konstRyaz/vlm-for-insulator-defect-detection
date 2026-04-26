# Final Stage 4 Package

This folder is the compact final artifact bundle for the leakage-free Stage 4 checkpoint.

The best current Stage 4 candidate is `context_030_maxpix401k`: predicted crop padding `0.30` with Qwen `max_pixels=401408`.

It improves the end-to-end object correctness from `21/58` to `23/58` compared with the tight predicted crop. The clean Stage 3 GT-crop ceiling is `27/58`, so the remaining gap is `4` objects.

Use `final_metrics_table.csv` for the headline table, and `stage4_context_comparison_report.md` for the comparison narrative. The PNG files are ready for visual inspection or presentation slides.
