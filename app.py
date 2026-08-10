from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from datapilot.analyst import dataframe_csv, gemini_dataset_summary, inspect_dataset
from datapilot.config import get_settings
from datapilot.data import SAMPLE_DATASETS, load_sample
from datapilot.workflow import run_analysis

st.set_page_config(
    page_title="DataPilot · Autonomous Data Analyst",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
:root{--navy:#07111f;--panel:#0e1b2c;--line:#203149;--cyan:#49d7c5;--blue:#6d8dff;--text:#edf4ff;--muted:#91a1b7}
.stApp{background:radial-gradient(circle at 75% -10%,#17355c 0,transparent 35%),#07111f;color:var(--text)}
html,body,[class*="css"]{font-family:"DM Sans",sans-serif}
h1,h2,h3{font-family:"Manrope",sans-serif;letter-spacing:-.03em}
header[data-testid="stHeader"]{background:transparent}
div[data-testid="stSidebar"]{background:#091522;border-right:1px solid var(--line)}
.block-container{max-width:1480px;padding-top:1.1rem;padding-bottom:4rem}
.brand{display:flex;gap:.75rem;align-items:center;font:800 1.2rem Manrope;color:white;margin:.2rem 0 1.3rem}
.brand-mark{display:grid;place-items:center;width:34px;height:34px;border-radius:10px;background:linear-gradient(135deg,var(--cyan),var(--blue));color:#07111f}
.hero{border:1px solid #29405d;background:linear-gradient(125deg,rgba(17,35,57,.96),rgba(9,22,38,.88));border-radius:24px;padding:2rem 2.2rem;margin-bottom:1rem;overflow:hidden;position:relative}
.hero:after{content:"";position:absolute;width:340px;height:340px;border-radius:50%;right:-100px;top:-190px;background:rgba(73,215,197,.10)}
.eyebrow{color:var(--cyan);font-size:.73rem;font-weight:700;letter-spacing:.18em;text-transform:uppercase}
.hero h1{font-size:clamp(2.1rem,4vw,4rem);line-height:1.02;margin:.45rem 0 .7rem;color:white}
.hero p{max-width:790px;color:#aebdd0;font-size:1.02rem;line-height:1.65;margin:0}
.stepbar{display:flex;gap:.5rem;flex-wrap:wrap;margin-top:1.4rem}.step{border:1px solid #2d4664;border-radius:999px;padding:.4rem .72rem;color:#9eafc4;font-size:.75rem}.step.on{color:#07111f;background:var(--cyan);border-color:var(--cyan);font-weight:700}
.panel{background:rgba(14,27,44,.92);border:1px solid var(--line);border-radius:18px;padding:1.15rem 1.25rem;height:100%}
.kicker{color:var(--cyan);font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.12em}.muted{color:var(--muted);font-size:.87rem;line-height:1.55}
.signature{background:linear-gradient(145deg,#11263b,#0b1828);border:1px solid #29435f;border-radius:17px;padding:1rem;margin-top:1rem}.signature strong{color:white}.signature a{color:var(--cyan);text-decoration:none;font-size:.83rem}
div[data-testid="stMetric"]{background:#0d1b2c;border:1px solid var(--line);padding:15px 17px;border-radius:15px}div[data-testid="stMetric"] label{color:#91a1b7}div[data-testid="stMetricValue"]{color:white}
.stButton>button,.stDownloadButton>button{border:0;border-radius:11px;background:linear-gradient(135deg,#49d7c5,#6d8dff);color:#07111f;font-weight:800}
.stButton>button:hover,.stDownloadButton>button:hover{color:#07111f;filter:brightness(1.08)}
div[data-testid="stFileUploaderDropzone"]{background:#0c1a2b;border:1.5px dashed #3b617c;border-radius:16px;padding:1.3rem}
div[data-baseweb="tab-list"]{gap:.3rem;background:#0b1828;border:1px solid var(--line);border-radius:13px;padding:.3rem}
button[data-baseweb="tab"]{border-radius:9px;color:#9caec3}button[data-baseweb="tab"][aria-selected="true"]{background:#172b41;color:white}
.stDataFrame{border:1px solid var(--line);border-radius:13px;overflow:hidden}
[data-testid="stAlert"]{border-radius:13px}
</style>
""",
    unsafe_allow_html=True,
)

settings = get_settings()
for key, default in {
    "frame": None,
    "dataset_name": "",
    "profile": None,
    "result": None,
    "ai_summary": "",
    "chat": [],
    "target": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def read_upload(uploaded) -> pd.DataFrame:
    suffix = Path(uploaded.name).suffix.lower()
    raw = uploaded.getvalue()
    if len(raw) > settings.max_upload_mb * 1_048_576:
        raise ValueError(f"File exceeds the {settings.max_upload_mb} MB limit.")
    stream = io.BytesIO(raw)
    if suffix in {".csv", ".tsv", ".txt"}:
        return pd.read_csv(stream, sep="\t" if suffix == ".tsv" else None, engine="python")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(stream)
    if suffix == ".parquet":
        return pd.read_parquet(stream)
    if suffix == ".json":
        try:
            return pd.read_json(stream)
        except ValueError:
            stream.seek(0)
            return pd.read_json(stream, lines=True)
    raise ValueError("Use CSV, TSV, Excel, JSON, or Parquet.")


with st.sidebar:
    st.markdown(
        '<div class="brand"><span class="brand-mark">✦</span>DataPilot</div>',
        unsafe_allow_html=True,
    )
    st.caption("AUTONOMOUS ANALYSIS WORKSPACE")
    st.markdown("##### Gemini intelligence")
    default_key = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
    api_key = st.text_input(
        "Gemini API key",
        value=default_key,
        type="password",
        help="Masked in the interface and not intentionally written to files or logs.",
    )
    model = st.selectbox("Model", ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"])
    st.caption("● AI ready" if api_key else "○ Local analysis mode")
    st.divider()
    st.markdown("##### Privacy controls")
    metadata_only = st.toggle(
        "Metadata-first AI",
        value=True,
        help="Send schema, aggregate statistics, and three redacted examples—not the full dataset.",
    )
    excluded = st.multiselect(
        "Exclude columns from AI",
        list(st.session_state.frame.columns) if st.session_state.frame is not None else [],
    )
    st.divider()
    if st.button("Reset workspace", width="stretch"):
        for key in ("frame", "profile", "result", "ai_summary", "chat", "target"):
            st.session_state[key] = (
                None
                if key in {"frame", "profile", "result", "target"}
                else ([] if key == "chat" else "")
            )
        st.rerun()
    st.markdown(
        """
    <div class="signature">
      <div class="kicker">Built & designed by</div>
      <strong>Dinesh Barri</strong><br>
      <span class="muted">AI Engineer · Data Scientist</span><br><br>
      <a href="https://github.com/dineshbarri">GitHub ↗</a>&nbsp;&nbsp;
      <a href="https://www.linkedin.com/in/dinesh-barri-7654b010b">LinkedIn ↗</a>
    </div>""",
        unsafe_allow_html=True,
    )

loaded = st.session_state.frame is not None
st.markdown(
    f"""
<section class="hero">
  <div class="eyebrow">Evidence-first autonomous data science</div>
  <h1>Your data. Explained.<br>Decisions, accelerated.</h1>
  <p>Upload a dataset and DataPilot immediately inspects its structure, surfaces quality risks,
  recommends analytical targets, creates interactive evidence, and prepares a leakage-safe
  machine-learning study—with Gemini available for grounded interpretation.</p>
  <div class="stepbar">
    <span class="step {"on" if loaded else ""}">01 · Connect</span>
    <span class="step {"on" if loaded else ""}">02 · Inspect</span>
    <span class="step {"on" if st.session_state.ai_summary else ""}">03 · Interpret</span>
    <span class="step {"on" if st.session_state.result else ""}">04 · Model</span>
    <span class="step {"on" if st.session_state.result else ""}">05 · Deliver</span>
  </div>
</section>""",
    unsafe_allow_html=True,
)

if not loaded:
    left, right = st.columns([1.35, 0.65], gap="large")
    with left:
        st.markdown('<div class="kicker">Start a new analysis</div>', unsafe_allow_html=True)
        st.subheader("Drop in your dataset")
        uploaded = st.file_uploader(
            "Upload dataset",
            type=["csv", "tsv", "txt", "xlsx", "xls", "json", "parquet"],
            label_visibility="collapsed",
        )
        st.caption("CSV · TSV · Excel · JSON · Parquet  |  Raw data remains in this session.")
        if uploaded:
            try:
                with st.status("DataPilot is inspecting your dataset…", expanded=True) as status:
                    st.write("Validating file structure")
                    frame = read_upload(uploaded)
                    st.write("Profiling columns, missingness, cardinality, and target candidates")
                    profile = inspect_dataset(frame)
                    st.session_state.frame = frame
                    st.session_state.profile = profile
                    st.session_state.dataset_name = uploaded.name
                    status.update(label="Dataset ready", state="complete")
                st.rerun()
            except Exception as exc:
                st.error(f"Upload could not be processed: {exc}")
    with right:
        st.markdown(
            '<div class="panel"><div class="kicker">Try it instantly</div><h3>Explore a trusted demo</h3><p class="muted">Load a complete classification or regression dataset and see the full analyst workflow.</p></div>',
            unsafe_allow_html=True,
        )
        demo = st.selectbox("Demo dataset", list(SAMPLE_DATASETS))
        if st.button("Load demo workspace", width="stretch"):
            frame, target, name = load_sample(SAMPLE_DATASETS[demo])
            st.session_state.frame, st.session_state.target = frame, target
            st.session_state.dataset_name = name
            st.session_state.profile = inspect_dataset(frame)
            st.rerun()
    st.stop()

frame: pd.DataFrame = st.session_state.frame
profile = st.session_state.profile or inspect_dataset(frame)
brief = profile["brief"]

metrics = st.columns(6)
metrics[0].metric("Rows", f"{brief.rows:,}")
metrics[1].metric("Columns", f"{brief.columns:,}")
metrics[2].metric("Numeric", brief.numeric)
metrics[3].metric("Categorical", brief.categorical)
metrics[4].metric("Missing cells", f"{brief.missing_cells:,}")
metrics[5].metric("Quality score", f"{profile['quality_score']}/100")

overview, quality, explore, ai_tab, model_tab, deliver = st.tabs(
    ["Overview", "Data quality", "Explore", "AI insights", "Model lab", "Deliver"]
)

with overview:
    st.subheader(st.session_state.dataset_name)
    st.caption(f"Dataset fingerprint {brief.fingerprint} · {brief.memory_mb:.2f} MB in memory")
    first, last, sample = st.tabs(["First 5 rows", "Last 5 rows", "Random sample"])
    first.dataframe(frame.head(), width="stretch", hide_index=True)
    last.dataframe(frame.tail(), width="stretch", hide_index=True)
    sample.dataframe(
        frame.sample(min(5, len(frame)), random_state=42), width="stretch", hide_index=True
    )
    st.markdown("#### Data dictionary")
    st.dataframe(
        profile["dictionary"].drop(columns=["issue_count"]), width="stretch", hide_index=True
    )

with quality:
    a, b = st.columns([0.75, 1.25])
    with a:
        st.markdown("#### Quality signals")
        st.metric("Duplicate rows", f"{brief.duplicate_rows:,}")
        st.metric("Completeness", f"{100 - brief.missing_cells / max(1, frame.size) * 100:.1f}%")
        flagged = profile["dictionary"].query("issue_count > 0")
        st.metric("Flagged columns", len(flagged))
        st.info("DataPilot reports evidence first. No rows or values are changed without approval.")
    with b:
        missing = profile["missing"][profile["missing"] > 0].sort_values()
        if len(missing):
            fig = px.bar(
                x=missing.values,
                y=missing.index,
                orientation="h",
                labels={"x": "Missing values", "y": "Column"},
                title="Missing values by column",
                color=missing.values,
                color_continuous_scale=["#49d7c5", "#6d8dff"],
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0e1b2c",
                plot_bgcolor="#0e1b2c",
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.success("No missing values detected.")
        if len(flagged):
            st.dataframe(flagged.drop(columns=["issue_count"]), width="stretch", hide_index=True)

with explore:
    numeric = profile["numeric"]
    if numeric:
        selected = st.selectbox("Explore a numerical feature", numeric)
        c1, c2 = st.columns(2)
        fig = px.histogram(
            frame,
            x=selected,
            marginal="box",
            title=f"Distribution of {selected}",
            color_discrete_sequence=["#49d7c5"],
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0e1b2c", plot_bgcolor="#0e1b2c")
        c1.plotly_chart(fig, width="stretch")
        if not profile["correlation"].empty:
            heat = px.imshow(
                profile["correlation"],
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale=["#1a2940", "#49d7c5", "#f4b860"],
                title="Numeric correlation map",
            )
            heat.update_layout(template="plotly_dark", paper_bgcolor="#0e1b2c")
            c2.plotly_chart(heat, width="stretch")
        else:
            c2.info("Add another numerical column to calculate correlations.")
        st.dataframe(frame[numeric].describe().T, width="stretch")
    else:
        st.info("This dataset has no numerical columns. Use the categorical overview below.")
    categories = profile["categorical"]
    if categories:
        selected_cat = st.selectbox("Explore a categorical feature", categories)
        counts = frame[selected_cat].astype(str).value_counts().head(20).reset_index()
        fig = px.bar(
            counts,
            x="count",
            y=selected_cat,
            orientation="h",
            title=f"Top values · {selected_cat}",
            color="count",
            color_continuous_scale=["#49d7c5", "#6d8dff"],
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1b2c",
            plot_bgcolor="#0e1b2c",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, width="stretch")

with ai_tab:
    st.markdown("#### Ask Gemini to interpret the computed evidence")
    st.caption(
        "AI interpretation based on dataset metadata and limited redacted samples. Verify against source documentation."
    )
    if not api_key:
        st.warning(
            "Enter a Gemini API key in the sidebar. Deterministic profiling remains fully available without AI."
        )
    if st.button("Generate AI analyst brief", disabled=not bool(api_key)):
        try:
            with st.status("Gemini is reviewing the evidence package…", expanded=True) as status:
                st.write("Redacting potential PII and excluded columns")
                st.write("Sending schema, aggregate statistics, and three examples")
                summary = gemini_dataset_summary(frame, profile, api_key, model, excluded)
                st.session_state.ai_summary = summary
                status.update(label="AI analyst brief ready", state="complete")
        except ValueError as exc:
            st.error(str(exc))
    if st.session_state.ai_summary:
        st.markdown(st.session_state.ai_summary)
    with st.expander("What may be sent to Gemini"):
        st.write(
            "Column metadata, aggregate statistics, target candidates, quality score, and up to three redacted example rows."
        )
        st.write(
            "Automatically excluded potential PII:",
            [
                c
                for c in frame.columns
                if any(
                    k in str(c).lower() for k in ("email", "phone", "address", "name", "account")
                )
            ]
            or "None detected",
        )

with model_tab:
    st.markdown("#### Confirm the analytical target")
    candidates = pd.DataFrame(profile["targets"])
    st.dataframe(candidates, width="stretch", hide_index=True)
    default_target = st.session_state.target or (
        profile["targets"][0]["column"] if profile["targets"] else frame.columns[-1]
    )
    target = st.selectbox(
        "Target column", list(frame.columns), index=list(frame.columns).index(default_target)
    )
    st.caption("DataPilot will not train supervised models until you confirm this selection.")
    if frame[target].nunique(dropna=True) < 2:
        st.error("The selected target has fewer than two observed values.")
    run = st.button(
        "Run autonomous model study",
        type="primary",
        disabled=frame[target].nunique(dropna=True) < 2,
    )
    if run:
        try:
            progress = st.progress(0, text="Preparing agent graph")
            progress.progress(12, text="Data Quality Agent · auditing risks")
            with st.spinner(
                "LangGraph agents are profiling, planning, training, evaluating, and explaining…"
            ):
                result = run_analysis(frame, target, st.session_state.dataset_name, settings)
            progress.progress(100, text="Analysis complete")
            st.session_state.result = result.model_dump(mode="json")
            st.success(
                "Model study completed with leakage-safe preprocessing and cross-validation."
            )
        except Exception as exc:
            st.error(f"Model study failed: {exc}")
    result = st.session_state.result
    if result:
        best = result["model_results"][0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Selected model", result["best_model"])
        c2.metric(
            "One-time test " + best["primary_metric"].replace("_", " ").title(),
            f"{best['final_test_score']:.3f}",
        )
        c3.metric("CV mean", f"{best['cross_validation_mean']:.3f}")
        results = pd.DataFrame(result["model_results"])
        fig = px.bar(
            results.sort_values("selection_score"),
            x="selection_score",
            y="name",
            orientation="h",
            color="selection_score",
            title="Training-CV model selection",
            color_continuous_scale=["#344b69", "#49d7c5"],
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1b2c",
            plot_bgcolor="#0e1b2c",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, width="stretch")
        st.dataframe(results, width="stretch", hide_index=True)
        st.markdown("#### Agent execution trace")
        st.dataframe(pd.DataFrame(result["trace"]), width="stretch", hide_index=True)

with deliver:
    st.markdown("#### Export your evidence")
    c1, c2 = st.columns(2)
    c1.download_button(
        "Download original dataset · CSV",
        dataframe_csv(frame),
        file_name=f"{Path(st.session_state.dataset_name).stem}_datapilot.csv",
        mime="text/csv",
        width="stretch",
    )
    c2.download_button(
        "Download data dictionary · CSV",
        dataframe_csv(profile["dictionary"].drop(columns=["issue_count"])),
        file_name="datapilot_data_dictionary.csv",
        mime="text/csv",
        width="stretch",
    )
    result = st.session_state.result
    if result:
        st.markdown("#### Model and report artifacts")
        columns = st.columns(min(4, len(result["artifacts"])))
        for column, (name, raw_path) in zip(columns, result["artifacts"].items(), strict=False):
            path = Path(raw_path)
            if path.exists():
                column.download_button(
                    name.replace("_", " ").title(),
                    path.read_bytes(),
                    file_name=path.name,
                    width="stretch",
                )
    else:
        st.info(
            "Run a model study to unlock the fitted pipeline, model card, metrics, and HTML report."
        )

st.caption(
    "DataPilot provides exploratory decision support. Predictive associations do not establish causality."
)
