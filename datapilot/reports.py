from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import joblib


def export_artifacts(
    run_id: str,
    state: dict[str, Any],
    run_directory: Path,
) -> dict[str, str]:
    run_directory.mkdir(parents=True, exist_ok=True)
    bundle = state["model_bundle"]
    summary = state["summary_payload"]

    pipeline_path = run_directory / "model_pipeline.joblib"
    joblib.dump(bundle.pipeline, pipeline_path)

    metrics_path = run_directory / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    model_card_path = run_directory / "MODEL_CARD.md"
    model_card_path.write_text(_model_card(summary), encoding="utf-8")

    report_path = run_directory / "analysis_report.html"
    report_path.write_text(_html_report(summary), encoding="utf-8")

    requirements_path = run_directory / "reproduction.json"
    requirements_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "random_state": state["settings"].random_state,
                "test_size": state["settings"].test_size,
                "target": summary["profile"]["target"],
                "task_type": summary["profile"]["task_type"],
                "best_model": summary["best_model"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "pipeline": str(pipeline_path),
        "metrics": str(metrics_path),
        "model_card": str(model_card_path),
        "report": str(report_path),
        "reproduction": str(requirements_path),
    }


def _model_card(run: dict[str, Any]) -> str:
    best = run["model_results"][0]
    profile = run["profile"]
    issues = (
        "\n".join(f"- {item['message']}" for item in run["quality_issues"]) or "- None detected"
    )
    return f"""# Model Card — {run["dataset_name"]}

## Model details

- Run ID: `{run["run_id"]}`
- Task: {profile["task_type"]}
- Target: `{profile["target"]}`
- Selected model: **{run["best_model"]}**
- Primary holdout metric: `{best["primary_metric"]} = {best["primary_score"]:.4f}`
- Training rows before split: {profile["rows"]:,}

## Intended use

Exploratory decision support and portfolio demonstration. Validate with domain-specific,
out-of-time data before any consequential or production use.

## Evaluation

```json
{json.dumps(best["metrics"], indent=2)}
```

## Data-quality observations

{issues}

## Explainability

Method: **{run["explainability"]["method"]}**. Importance values are predictive associations,
not evidence of causation.

## Limitations

- Results depend on the uploaded dataset and chosen target.
- Automated task inference can be wrong; a domain owner should confirm the objective.
- Fairness, privacy, and legal review are outside the automatic approval gate.
"""


def _html_report(run: dict[str, Any]) -> str:
    best = run["model_results"][0]
    summary_items = "".join(f"<li>{html.escape(item)}</li>" for item in run["executive_summary"])
    recommendations = "".join(f"<li>{html.escape(item)}</li>" for item in run["recommendations"])
    issues = (
        "".join(
            f"<tr><td>{html.escape(item['severity'])}</td><td>{html.escape(item['code'])}</td>"
            f"<td>{html.escape(item['message'])}</td></tr>"
            for item in run["quality_issues"]
        )
        or "<tr><td colspan='3'>No material flags</td></tr>"
    )
    metrics = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{value:.4f}</td></tr>"
        for name, value in best["metrics"].items()
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>DataPilot AI report</title>
<style>
body{{font-family:Inter,system-ui,sans-serif;max-width:1000px;margin:40px auto;padding:0 24px;color:#172033}}
h1{{color:#5537d8}} .hero{{background:#f4f1ff;border:1px solid #d9d0ff;padding:24px;border-radius:18px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin:20px 0}}
.card{{border:1px solid #e2e6ef;border-radius:14px;padding:18px}} table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid #e2e6ef;padding:10px;text-align:left}} small{{color:#667085}}
</style></head><body>
<div class="hero"><h1>DataPilot AI Analysis Report</h1>
<p>{html.escape(run["dataset_name"])} · Run {html.escape(run["run_id"])}</p></div>
<div class="grid">
<div class="card"><small>Selected model</small><h2>{html.escape(run["best_model"])}</h2></div>
<div class="card"><small>{html.escape(best["primary_metric"])}</small><h2>{best["primary_score"]:.3f}</h2></div>
<div class="card"><small>Rows analyzed</small><h2>{run["profile"]["rows"]:,}</h2></div>
</div>
<h2>Executive findings</h2><ul>{summary_items}</ul>
<h2>Evaluation</h2><table><tr><th>Metric</th><th>Value</th></tr>{metrics}</table>
<h2>Data quality</h2><table><tr><th>Severity</th><th>Code</th><th>Observation</th></tr>{issues}</table>
<h2>Recommendations</h2><ol>{recommendations}</ol>
<p><small>Generated from computed evidence. Predictive findings do not establish causality.</small></p>
</body></html>"""
