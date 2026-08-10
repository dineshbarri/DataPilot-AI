# Reproducible smoke benchmarks

These results are correctness-oriented smoke benchmarks, not claims about production accuracy.
They were measured on a local Windows CPU with Python 3.13.1, `RANDOM_STATE=42`,
`OPTUNA_TRIALS=0`, and `MAX_CRITIC_RETRIES=0`.

| Dataset | Rows | Task | Selected model | Training CV score | Untouched test score | Wall time |
|---|---:|---|---|---:|---:|---:|
| scikit-learn Iris | 150 | multiclass classification | Logistic Regression | 0.9583 weighted F1 | 0.9333 weighted F1 | 4.29 s |
| scikit-learn Diabetes | 442 | regression | Linear Regression | 0.4493 R² | 0.4526 R² | 3.17 s |

Candidate selection uses cross-validation on the training partition only. The test score is
computed once after selection. Timings vary by hardware and installed optional dependencies.
Run the same path with `python scripts/benchmark.py` once optional benchmark automation is added.

