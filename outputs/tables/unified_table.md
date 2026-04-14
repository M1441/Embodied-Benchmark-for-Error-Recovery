# Unified Results Table

| Method | N | Acc (%) | 95% CI | Macro-F1 | Fmt% | F1 | F2 | F3 | F4 | F5 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| B: LookAround | 300 | 20.0 | [15.7, 24.7] | 0.0083 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| B: Blind-Heuristic | 300 | 37.3 | [32.0, 43.0] | 0.0416 | 100.0 | 86.7 | 0.0 | 0.0 | 100.0 | 0.0 |
| B: Heuristic-Oracle | 300 | 33.7 | [28.3, 39.3] | 0.0275 | 100.0 | 100.0 | 8.3 | 1.7 | 58.3 | 0.0 |
| B: Oracle | 300 | 100.0 | [100.0, 100.0] | 1.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| V: gpt-5.4 text-only | 300 | 16.7 | [12.7, 21.0] | 0.1162 | 99.3 | 8.3 | 25.0 | 13.3 | 25.0 | 11.7 |
| V: gpt-5.4 after-only | 300 | 24.7 | [20.3, 29.7] | 0.0913 | 99.3 | 53.3 | 21.7 | 16.7 | 28.3 | 3.3 |
| V: gpt-5.4 full (B+A) | 300 | 28.3 | [23.7, 33.3] | 0.1199 | 100.0 | 61.7 | 21.7 | 18.3 | 31.7 | 8.3 |
| V: gpt-5.4 swap-test | 300 | 28.3 | [23.7, 33.3] | 0.1074 | 100.0 | 48.3 | 26.7 | 20.0 | 40.0 | 6.7 |
| V: gpt-5.4 full+counterfactual | 300 | 0.7 | [0.0, 1.7] | 0.0003 | 99.3 | 3.3 | 0.0 | 0.0 | 0.0 | 0.0 |
| V: gpt-5.4 full+mental_simulation | 300 | 26.7 | [22.0, 31.7] | 0.1054 | 100.0 | 73.3 | 11.7 | 36.7 | 5.0 | 6.7 |

## Semi-Loop Recovery (VLM only)

| Method | Recovery@K | Completion | Note |
|---|---:|---:|---|
| V: gpt-5.4 text-only | 10.0 | 10.0 |  |
| V: gpt-5.4 after-only | 8.3 | 8.3 |  |
| V: gpt-5.4 full (B+A) | 9.7 | 9.7 |  |
| V: gpt-5.4 swap-test | 10.7 | 10.7 |  |
| V: gpt-5.4 full+counterfactual | 0.0 | 0.0 |  |
| V: gpt-5.4 full+mental_simulation | 11.0 | 11.0 |  |

> **Note**: Recovery Rate ceiling is ~60% because F1 (LookAround) and F4 (Retry) have empty goal states that cannot be verified by the symbolic executor.
