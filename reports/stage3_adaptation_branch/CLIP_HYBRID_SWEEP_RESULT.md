# Stage 3 CLIP Hybrid Sweep Result

Run source: Kaggle kernel `kostyaryazanov/stage3-clip-hybrid-sweep-clean`, version 1.

This was an exploratory coarse-only sweep over CLIP image embeddings, logistic-regression regularization, class weighting, simple flashover thresholds, and a zero-shot broken rescue heuristic.

Important caveat: this sweep compares many variants on the same validation slice. It should be read as a hypothesis generator, not as a final tuned validation result.

## Best Variant

Best by three-class macro-F1:

`logreg_c0.03_balanced`

Metrics on the 58-object clean val slice:

- accuracy: `0.4828`
- macro-F1: `0.4799`
- insulator_ok recall: `0.2813`
- defect_flashover recall: `0.8500`
- defect_broken recall: `0.3333`

Confusion matrix:

```text
,insulator_ok,defect_flashover,defect_broken
insulator_ok,9,23,0
defect_flashover,3,17,0
defect_broken,1,3,2
```

## Interpretation

The sweep confirms that CLIP image embeddings contain a strong flashover signal. The best macro-F1 variant correctly recovers `17/20` flashover cases and `2/6` broken cases, which is better class balance than the frozen Qwen variants. The cost is severe overcalling of flashover on normal insulators: `23/32` `insulator_ok` samples become `defect_flashover`.

This is not yet a final classifier. It is a strong reason to build a proper train-only hybrid classifier experiment with a held-out validation rule, or to use CLIP embeddings as an auxiliary feature rather than a direct replacement for Qwen.

## Decision

Keep the hybrid branch alive. The next clean version should avoid selecting thresholds on the same val slice. Use train-only model selection, cross-validation on train, or a small frozen rule chosen before touching val.
