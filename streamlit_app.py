from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from datapilot.config import get_settings
from datapilot.data import SAMPLE_DATASETS, load_sample, read_dataset
from datapilot.insights import answer_follow_up
from datapilot.workflow import run_analysis

st.set_page_config(
    page_title="DataPilot AI",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root{--ink:#172033;--violet:#6d4aff;--mint:#22c5a5;--soft:#f6f5ff}
.stApp{background:linear-gradient(180deg,#fbfaff 0%,#ffffff 38%)}
.hero{padding:2.2rem 2.4rem;border-radius:24px;background:
linear-gradient(135deg,#21154f 0%,#5638c9 56%,#19a98e 140%);color:white;margin-bottom:1.25rem;
box-shadow:0 18px 42px rgba(77,53,168,.20)}
.hero h1{font-size:3.1rem;line-height:1.02;margin:0 0 .65rem;font-weight:800;letter-spacing:-.04em}
.hero p{font-size:1.08rem;max-width:820px;color:#eeeaff;margin:0}
.eyebrow{font-size:.78rem;text-transform:uppercase;letter-spacing:.15em;color:#cfc5ff;font-weight:700}
.trust{display:flex;gap:.6rem;flex-wrap:wrap;margin-top:1.15rem}
.pill{border:1px solid rgba(255,255,255,.25);background:rgba(255,255,255,.10);
padding:.38rem .68rem;border-radius:999px;font-size:.78rem}
.section-title{font-size:1.45rem;font-weight:750;color:var(--ink);margin:.3rem 0 .85rem}
.signal{border:1px solid #e7e3fa;border-radius:16px;padding:1rem;background:white;height:100%}
.signal strong{color:#5638c9}.small-muted{font-size:.82rem;color:#667085}
div[data-testid="stMetric"]{background:#fff;border:1px solid #e9e7f2;padding:15px 18px;border-radius:16px;
box-shadow:0 7px 18px rgba(34,24,72,.05)}
div[data-testid="stSidebar"]{background:#f4f1ff}
.stButton>button{border-radius:12px;font-weight:700;border:0;background:#6045d8;color:white}
.stButton>button:hover{background:#4f36c1;color:white}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <div class="eyebrow">Autonomous Data Science · Evidence First</div>
  <h1>DataPilot AI</h1>
  <p>From raw CSV or Parquet to validated models, explainable predictions, and a reproducible
  decision report—coordinated by a stateful LangGraph agent team.</p>
  <div class="trust">
    <span class="pill">No API key required</span><span class="pill">Leakage-safe pipelines</span>
    <span class="pill">Critic retry loop</span><span class="pill">SHAP-ready explanations</span>
    <span class="pill">Exportable model card</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

settings = get_settings()

with st.sidebar:
    st.header("Launch an analysis")
    source_mode = st.radio("Data source", ["Built-in demo", "Upload dataset"])
    frame: pd.DataFrame | None = None
    dataset_name = ""
    target = ""
    if source_mode == "Built-in demo":
        sample_label = st.selectbox("Demo dataset", list(SAMPLE_DATASETS))
        sample_key = SAMPLE_DATASETS[sample_label]
        frame, target, dataset_name = load_sample(sample_key)
        st.caption(f"{len(frame):,} rows · {len(frame.columns):,} columns · target: `{target}`")
    else:
        uploaded = st.file_uploader("CSV or Parquet", type=["csv", "parquet"])
        if uploaded is not None:
            try:
                frame = read_dataset(uploaded.getvalue(), uploaded.name, settings)
                dataset_name = uploaded.name
                target = st.selectbox("Target column", frame.columns, index=len(frame.columns) - 1)
                st.caption(f"{len(frame):,} rows · {len(frame.columns):,} columns")
            except ValueError as exc:
                st.error(str(exc))
    run_clicked = st.button(
        "Run autonomous analysis",
        type="primary",
        width="stretch",
        disabled=frame is None,
    )
    st.divider()
    st.caption(
        "Upload limits protect the public demo. Generated Python is never executed in this web process."
    )

if "result" not in st.session_state:
    st.session_state.result = None

if run_clicked and frame is not None:
    progress = st.progress(0, text="Starting agent workflow…")
    try:
        progress.progress(15, text="Profiling data and auditing quality…")
        with st.spinner("Agents are analyzing, validating, modeling, and explaining the dataset…"):
            result = run_analysis(frame, target, dataset_name, settings)
        progress.progress(100, text="Analysis complete")
        st.session_state.result = result.model_dump(mode="json")
        st.success("Analysis completed and reproducible artifacts were generated.")
    except Exception as exc:
        progress.empty()
        st.error(f"Analysis failed: {exc}")

result = st.session_state.result
if result is None:
    st.markdown(
        '<div class="section-title">What the agent team delivers</div>', unsafe_allow_html=True
    )
    columns = st.columns(4)
    promises = [
        (
            "01",
            "Trust the data",
            "Missingness, imbalance, duplicates, outliers, leakage risk, and drift-ready checks.",
        ),
        (
            "02",
            "Compare honestly",
            "Preprocessing stays inside each fitted pipeline; holdout and CV scores remain separate.",
        ),
        (
            "03",
            "Challenge results",
            "A critic agent rejects weak or unstable models and routes the graph back for a retry.",
        ),
        (
            "04",
            "Ship evidence",
            "Interactive charts, model card, fitted pipeline, metrics JSON, and an HTML decision report.",
        ),
    ]
    for column, (number, title, body) in zip(columns, promises, strict=False):
        with column:
            st.markdown(
                f'<div class="signal"><strong>{number}</strong><h3>{title}</h3>'
                f'<div class="small-muted">{body}</div></div>',
                unsafe_allow_html=True,
            )
    st.info("Choose a built-in dataset and click **Run autonomous analysis** for the fastest demo.")
    st.stop()

profile = result["profile"]
best = result["model_results"][0]
metrics = st.columns(5)
metrics[0].metric("Rows analyzed", f"{profile['rows']:,}")
metrics[1].metric("Features", f"{profile['columns'] - 1:,}")
metrics[2].metric("Best model", result["best_model"])
metrics[3].metric(best["primary_metric"].replace("_", " ").title(), f"{best['primary_score']:.3f}")
metrics[4].metric("Quality flags", len(result["quality_issues"]))

overview_tab, model_tab, explain_tab, workflow_tab, artifacts_tab = st.tabs(
    ["Executive overview", "Model laboratory", "Explainability", "Agent trace", "Artifacts & Q&A"]
)

with overview_tab:
    left, right = st.columns([1.1, 0.9])
    with left:
        st.subheader("Evidence-backed findings")
        for item in result["executive_summary"]:
            st.markdown(f"- {item}")
        st.subheader("Recommended next actions")
        for index, item in enumerate(result["recommendations"], start=1):
            st.markdown(f"{index}. {item}")
    with right:
        st.subheader("Data-quality observations")
        if result["quality_issues"]:
            quality_frame = pd.DataFrame(result["quality_issues"])[
                ["severity", "code", "column", "message"]
            ]
            st.dataframe(quality_frame, hide_index=True, width="stretch")
        else:
            st.success("No material quality flags were detected.")
        with st.expander("Analysis plan"):
            st.json(result["plan"])

with model_tab:
    model_frame = pd.DataFrame(result["model_results"])
    figure = px.bar(
        model_frame.sort_values("primary_score"),
        x="primary_score",
        y="name",
        orientation="h",
        color="primary_score",
        color_continuous_scale=["#d9d0ff", "#5f43d6", "#1ca58c"],
        labels={"primary_score": best["primary_metric"], "name": "Candidate model"},
        title=f"Candidate model comparison — {best['primary_metric'].replace('_', ' ')}",
    )
    figure.update_layout(coloraxis_showscale=False, height=430, margin=dict(l=10, r=20, t=60, b=20))
    st.plotly_chart(figure, width="stretch")
    display = model_frame[
        [
            "name",
            "primary_score",
            "cross_validation_mean",
            "cross_validation_std",
            "training_seconds",
        ]
    ]
    st.dataframe(display, hide_index=True, width="stretch")
    critic = result["critic"]
    if critic["approved"]:
        st.success("Critic decision: approved. " + " ".join(critic["reasons"]))
    else:
        st.warning("Critic decision: rejected. " + " ".join(critic["reasons"]))

with explain_tab:
    importance = result["explainability"]["feature_importance"]
    importance_frame = pd.DataFrame(
        [{"feature": key, "importance": value} for key, value in importance.items()]
    )
    figure = px.bar(
        importance_frame.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=["#e9e5ff", "#6848dd"],
        title=f"Predictive feature importance — {result['explainability']['method']}",
    )
    figure.update_layout(coloraxis_showscale=False, height=500, margin=dict(l=10, r=20, t=60, b=20))
    st.plotly_chart(figure, width="stretch")
    for caveat in result["explainability"]["caveats"]:
        st.caption("⚠️ " + caveat)

with workflow_tab:
    trace_frame = pd.DataFrame(result["trace"])
    st.dataframe(trace_frame, hide_index=True, width="stretch")
    st.markdown(
        "The **Evaluation / Critic Agent** conditionally routes back to the **Modeling Agent** "
        "when the score or validation consistency misses the quality gate."
    )

with artifacts_tab:
    st.subheader("Download reproducible outputs")
    artifact_columns = st.columns(min(5, len(result["artifacts"])))
    for column, (name, raw_path) in zip(
        artifact_columns, result["artifacts"].items(), strict=False
    ):
        path = Path(raw_path)
        if path.exists():
            column.download_button(
                name.replace("_", " ").title(),
                data=path.read_bytes(),
                file_name=path.name,
                mime="application/octet-stream",
                width="stretch",
            )
    st.subheader("Ask this analysis")
    question = st.text_input(
        "Question",
        placeholder="Which model won, what are the quality risks, or which features matter?",
    )
    if question:
        st.markdown(answer_follow_up(result, question))
    with st.expander("Evidence registry"):
        st.dataframe(pd.DataFrame(result["evidence"]), hide_index=True, width="stretch")
    with st.expander("Run JSON"):
        st.code(json.dumps(result, indent=2), language="json")
