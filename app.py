from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results"
REPORT_PATHS = [ROOT_DIR / "report.json", RESULTS_DIR / "report.json"]
PROJECT_TITLE = "Obesity Disease Risk ML"


st.set_page_config(
    page_title=PROJECT_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --accent: #146c67;
            --accent-soft: #e7f3f1;
            --ink: #17202a;
            --muted: #536171;
            --surface: #ffffff;
            --surface-soft: #f6f8fa;
            --line: #d9e0e6;
        }

        .stApp {
            background: #ffffff;
            color: var(--ink);
        }

        .block-container {
            padding-top: 2.1rem;
            padding-bottom: 3rem;
            max-width: 1240px;
        }

        [data-testid="stSidebar"] {
            background: #f7faf9;
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"],
        [data-testid="stSidebar"] *,
        .stApp,
        .stApp p,
        .stApp label,
        .stApp span,
        .stApp div {
            color: var(--ink);
        }

        h1, h2, h3, h4 {
            color: var(--ink);
            letter-spacing: 0;
        }

        h1 {
            font-size: 2.1rem;
            line-height: 1.08;
            margin-bottom: 0.35rem;
        }

        h2 {
            margin-top: 0.8rem;
        }

        .lede {
            color: var(--muted);
            font-size: 1.05rem;
            line-height: 1.55;
            max-width: 920px;
        }

        .metric-note {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.45;
        }

        .section-note {
            color: var(--muted);
            margin-top: -0.35rem;
            margin-bottom: 0.8rem;
        }

        [data-testid="stMetric"] {
            background: var(--surface-soft);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.85rem 0.9rem;
        }

        [data-testid="stMetric"] label,
        [data-testid="stMetric"] [data-testid="stMetricLabel"],
        [data-testid="stMetric"] [data-testid="stMetricValue"],
        [data-testid="stMetric"] [data-testid="stMetricDelta"] {
            color: var(--ink);
        }

        [data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color: var(--muted);
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
        }

        div[data-testid="stDataFrame"] * {
            color: inherit;
        }

        div[data-baseweb="radio"] label,
        div[data-baseweb="checkbox"] label,
        div[data-baseweb="select"] label {
            color: var(--ink);
        }

        button[kind="secondary"] {
            color: var(--ink);
            border-color: var(--line);
            background: var(--surface);
        }

        hr {
            border-color: var(--line);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_report() -> dict:
    for path in REPORT_PATHS:
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
    return {}


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def fmt_num(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def fmt_int(value: float | int) -> str:
    return f"{int(round(value)):,}"


def model_table(models: dict, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for model_name, values in models.items():
        row = {"Model": model_name}
        for metric in metrics:
            row[metric.upper()] = values[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def image_grid(images: list[tuple[str, str]]) -> None:
    for row_start in range(0, len(images), 2):
        cols = st.columns(2)
        for col, (caption, filename) in zip(cols, images[row_start : row_start + 2]):
            path = RESULTS_DIR / filename
            with col:
                if path.exists():
                    st.image(str(path), caption=caption, use_container_width=True)
                else:
                    st.info(f"Missing visualization: `{filename}`")


def scenario_risk(
    baseline_rate: float,
    bmi: float,
    age: int,
    glucose: float,
    hypertension: bool,
    smoker: bool,
) -> float:
    risk = baseline_rate

    if bmi >= 30:
        risk *= 1.18
    elif bmi < 18.5:
        risk *= 1.04
    elif bmi < 25:
        risk *= 0.92

    if age >= 70:
        risk *= 2.15
    elif age >= 55:
        risk *= 1.55
    elif age < 35:
        risk *= 0.55

    if glucose >= 180:
        risk *= 1.45
    elif glucose >= 130:
        risk *= 1.2

    if hypertension:
        risk *= 1.35

    if smoker:
        risk *= 1.18

    return min(risk, 0.65)


def render_header(report: dict) -> None:
    st.title("Obesity and Disease Risk Prediction")
    st.markdown(
        """
        <div class="lede">
        Interactive frontend for the ML study: clinical disease-risk analysis,
        obesity effect estimates, model comparison, and generated results.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if report:
        cols = st.columns(3)
        dataset = report["dataset"]
        effect = report["obesity_effect_on_disease_risk"]

        cols[0].metric("Clinical samples", fmt_int(dataset["total_samples"]))
        cols[1].metric("Dataset obesity rate", fmt_pct(dataset["obesity_rate"]))
        cols[2].metric("Heart disease RR", f"{fmt_num(effect['heart_disease_relative_risk'])}x")


def render_overview(report: dict) -> None:
    dataset = report["dataset"]
    effect = report["obesity_effect_on_disease_risk"]

    st.subheader("Study Snapshot")
    cols = st.columns(3)
    cols[0].metric("Stroke rate", fmt_pct(dataset["stroke_rate"]))
    cols[1].metric("Heart disease rate", fmt_pct(dataset["heart_disease_rate"]))
    cols[2].metric("Heart disease relative risk", f"{fmt_num(effect['heart_disease_relative_risk'])}x")

    st.markdown(
        """
        <p class="section-note">
        The dashboard reads directly from the generated project artifacts, so updated
        reports and plots will flow into the app without rewriting the UI.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Individual Disease Risk")
    risk_df = pd.DataFrame(
        [
            {
                "Outcome": "Stroke",
                "Obese rate": effect["stroke_rate_obese"],
                "Non-obese rate": effect["stroke_rate_nonobese"],
                "Relative risk": effect["stroke_relative_risk"],
            },
            {
                "Outcome": "Heart disease",
                "Obese rate": effect["heart_disease_rate_obese"],
                "Non-obese rate": effect["heart_disease_rate_nonobese"],
                "Relative risk": effect["heart_disease_relative_risk"],
            },
        ]
    )
    st.dataframe(
        risk_df.style.format(
            {
                "Obese rate": "{:.2%}",
                "Non-obese rate": "{:.2%}",
                "Relative risk": "{:.2f}x",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Generated Visual Evidence")
    image_grid(
        [
            ("Disease rates across BMI groups", "bmi_disease_risk.png"),
            ("Obesity effect summary", "obesity_effect_summary.png"),
            ("Feature importance", "feature_importance.png"),
        ]
    )


def render_risk_lab(report: dict) -> None:
    st.subheader("Clinical Risk Scenario")
    st.markdown(
        """
        <p class="section-note">
        This panel is an exploratory scenario tool based on aggregate study rates.
        It is not a medical diagnosis and does not replace a trained, persisted model.
        </p>
        """,
        unsafe_allow_html=True,
    )

    effect = report["obesity_effect_on_disease_risk"]
    dataset = report["dataset"]

    controls, results = st.columns([0.9, 1.1])
    with controls:
        age = st.slider("Age", min_value=1, max_value=100, value=54)
        bmi = st.slider("BMI", min_value=12.0, max_value=60.0, value=31.0, step=0.1)
        glucose = st.slider("Average glucose level", min_value=50.0, max_value=280.0, value=125.0, step=1.0)
        hypertension = st.toggle("Hypertension", value=False)
        smoker = st.toggle("Current or former smoker", value=False)

    stroke_base = effect["stroke_rate_obese"] if bmi >= 30 else effect["stroke_rate_nonobese"]
    heart_base = effect["heart_disease_rate_obese"] if bmi >= 30 else effect["heart_disease_rate_nonobese"]
    stroke_est = scenario_risk(stroke_base, bmi, age, glucose, hypertension, smoker)
    heart_est = scenario_risk(heart_base, bmi, age, glucose, hypertension, smoker)
    bmi_status = "Obese" if bmi >= 30 else "Non-obese"

    with results:
        cols = st.columns(3)
        cols[0].metric("BMI status", bmi_status)
        cols[1].metric("Scenario stroke risk", fmt_pct(stroke_est, 2), delta=f"Study base {fmt_pct(dataset['stroke_rate'], 2)}")
        cols[2].metric(
            "Scenario heart disease risk",
            fmt_pct(heart_est, 2),
            delta=f"Study base {fmt_pct(dataset['heart_disease_rate'], 2)}",
        )

        comparison = pd.DataFrame(
            [
                {"Outcome": "Stroke", "Study average": dataset["stroke_rate"], "Scenario": stroke_est},
                {"Outcome": "Heart disease", "Study average": dataset["heart_disease_rate"], "Scenario": heart_est},
            ]
        ).set_index("Outcome")
        st.bar_chart(comparison)
        st.markdown(
            '<p class="metric-note">For true patient-level prediction, the next upgrade is to persist the trained preprocessing pipeline and model artifacts with joblib.</p>',
            unsafe_allow_html=True,
        )


def render_model_performance(report: dict) -> None:
    st.subheader("Model Performance")

    disease_tabs = st.tabs(["Stroke", "Heart disease"])
    with disease_tabs[0]:
        stroke = report["individual_risk_prediction"]["stroke"]
        st.metric("Best model", stroke["best_model"], delta=f"AUC {stroke['best_auc']:.3f}")
        df = model_table(stroke["all_models"], ["accuracy", "precision", "recall", "f1", "auc"])
        st.dataframe(df.style.format({col: "{:.3f}" for col in df.columns if col != "Model"}), use_container_width=True, hide_index=True)
        st.bar_chart(df.set_index("Model")[["AUC", "RECALL", "F1"]])

    with disease_tabs[1]:
        heart = report["individual_risk_prediction"]["heart_disease"]
        st.metric("Best model", heart["best_model"], delta=f"AUC {heart['best_auc']:.3f}")
        df = model_table(heart["all_models"], ["accuracy", "precision", "recall", "f1", "auc"])
        st.dataframe(df.style.format({col: "{:.3f}" for col in df.columns if col != "Model"}), use_container_width=True, hide_index=True)
        st.bar_chart(df.set_index("Model")[["AUC", "RECALL", "F1"]])

    st.subheader("Saved Comparison Plots")
    image_grid(
        [
            ("Clinical model comparison", "model_comparison.png"),
        ]
    )


def render_artifacts(report: dict) -> None:
    st.subheader("Project Artifacts")
    files = [
        ("Report JSON", ROOT_DIR / "report.json"),
        ("Notebook", ROOT_DIR / "obesity_disease_transmission.ipynb"),
        ("Pipeline script", ROOT_DIR / "obesity_disease_transmission_backup.py"),
        ("Results notes", ROOT_DIR / "RESULTS.md"),
        ("README", ROOT_DIR / "README.md"),
    ]
    rows = []
    for label, path in files:
        rows.append(
            {
                "Artifact": label,
                "Path": str(path.relative_to(ROOT_DIR)),
                "Available": path.exists(),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Feature Set")
    st.dataframe(pd.DataFrame({"Feature": report["dataset"]["features"]}), use_container_width=True, hide_index=True)

    st.subheader("Raw Report")
    st.json(report, expanded=False)


def main() -> None:
    inject_styles()
    report = load_report()
    if not report:
        st.error("Could not find `report.json`. Run the ML pipeline first or place the report in the project root.")
        st.stop()

    render_header(report)

    page = st.sidebar.radio(
        "Navigate",
        [
            "Overview",
            "Clinical risk scenario",
            "Model performance",
            "Artifacts",
        ],
    )

    st.sidebar.divider()
    st.sidebar.caption("Data source")
    st.sidebar.write(report["dataset"]["name"])
    st.sidebar.caption("Project")
    st.sidebar.write(PROJECT_TITLE)

    if page == "Overview":
        render_overview(report)
    elif page == "Clinical risk scenario":
        render_risk_lab(report)
    elif page == "Model performance":
        render_model_performance(report)
    else:
        render_artifacts(report)


if __name__ == "__main__":
    main()
