# Stage 4 Paired Analysis

Baseline: `qwen_stage4_context_pad030`
Candidate: `dinov2_qwen_secondbest_cv035`

| metric | value |
|---|---:|
| total | 58 |
| baseline correct | 23 |
| candidate correct | 34 |
| delta correct | 11 |
| helped | 21 |
| hurt | 10 |
| sign-test p | 0.0708 |
| bootstrap delta 95% CI | [0.0000, 0.3793] |

Interpretation: the sign test uses only changed cases. The bootstrap interval is over paired GT objects and should be read cautiously on a 58-object validation slice.
