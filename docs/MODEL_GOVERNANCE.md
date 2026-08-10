# Model Governance

DataPilot AI is an exploratory copilot. Its critic gate evaluates predictive performance and
validation consistency; it does not replace domain approval.

## Automatic gate

- Primary classification metric: balanced accuracy.
- Primary regression metric: R².
- Holdout score compared with a configurable minimum.
- Holdout-to-cross-validation divergence above 0.20 is flagged.
- A failed gate routes once back to modeling by default.
- When the retry budget is exhausted, the result is retained only with an explicit limitation.

## Human gate before deployment

- Confirm the target is meaningful and available at prediction time.
- Remove direct and proxy leakage.
- Evaluate out-of-time and segment performance.
- Review fairness and disparate impact.
- Verify privacy, consent, retention and lawful use.
- Establish drift, quality and performance alerts.
- Define rollback and retraining ownership.

## Reproducibility

Every completed run exports:

- serialized fitted pipeline
- metrics and evidence JSON
- model card
- standalone HTML analysis report
- reproduction metadata containing seed, split, target, task and selected model

