# AI and ML evaluation

## Leakage-safe ML protocol

1. Split the untouched test partition once using the configured random seed.
2. Tune and compare candidates using cross-validation on the training partition only.
3. Rank candidates by CV mean and expose CV standard deviation.
4. Refit the selected configuration on all training data.
5. Evaluate the selected pipeline exactly once on the untouched test partition.
6. Persist selection and final-test metrics separately.

Time-ordered or grouped production data must use the corresponding split strategy; the default
random split is not a substitute for domain-aware validation.

## LLM evaluation

`datapilot.evaluation` provides deterministic gates for evidence-ID validity, unsupported numeric
claims, PII leakage, and prompt-injection echoing. Golden cases should be expanded and evaluated
for every prompt or Gemini model change. Each production run should record prompt version, model
version, evidence payload hash, latency, token usage, estimated cost, and validation outcome.

## Initial quality gates

| Gate | Initial target |
|---|---:|
| Unit/integration coverage | 75%, rising to 85% |
| Unsupported numeric claims | 0 |
| Invalid evidence IDs | 0 |
| PII leakage | 0 |
| Prompt-injection compliance | 0 |
