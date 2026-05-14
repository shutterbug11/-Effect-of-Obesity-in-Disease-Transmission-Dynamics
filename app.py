from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results"
REPORT_PATHS = [ROOT_DIR / "report.json", RESULTS_DIR / "report.json"]


st.set_page_config(
    page_title="Obesity and Disease Transmission ML",
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


@st.cache_data
def simulate_epidemic(
    population_size: int,
    days: int,
    obesity_rate: float,
    initial_infected: int,
    beta: float,
    gamma: float,
    obesity_multiplier: float,
    contacts_per_day: float,
) -> pd.DataFrame:
    obese_total = max(1.0, population_size * obesity_rate)
    nonobese_total = max(1.0, population_size - obese_total)

    obese_weight = obese_total * obesity_multiplier
    nonobese_weight = nonobese_total
    obese_initial = min(obese_total, initial_infected * obese_weight / (obese_weight + nonobese_weight))
    nonobese_initial = min(nonobese_total, initial_infected - obese_initial)

    s_obese = obese_total - obese_initial
    i_obese = obese_initial
    r_obese = 0.0
    s_nonobese = nonobese_total - nonobese_initial
    i_nonobese = nonobese_initial
    r_nonobese = 0.0

    rows = []
    for day in range(days + 1):
        total_i = i_obese + i_nonobese
        rows.append(
            {
                "day": day,
                "susceptible": s_obese + s_nonobese,
                "infected": total_i,
                "recovered": r_obese + r_nonobese,
                "new_infections": 0.0 if day == 0 else new_obese + new_nonobese,
                "new_infections_obese": 0.0 if day == 0 else new_obese,
                "new_infections_nonobese": 0.0 if day == 0 else new_nonobese,
                "total_obese_infected": i_obese + r_obese,
                "total_nonobese_infected": i_nonobese + r_nonobese,
            }
        )

        if day == days:
            break

        infection_pressure = beta * contacts_per_day * total_i / population_size
        new_nonobese = s_nonobese * (1 - np.exp(-infection_pressure))
        new_obese = s_obese * (1 - np.exp(-infection_pressure * obesity_multiplier))

        recover_obese = gamma * i_obese
        recover_nonobese = gamma * i_nonobese

        s_obese = max(0.0, s_obese - new_obese)
        i_obese = max(0.0, i_obese + new_obese - recover_obese)
        r_obese = min(obese_total, r_obese + recover_obese)

        s_nonobese = max(0.0, s_nonobese - new_nonobese)
        i_nonobese = max(0.0, i_nonobese + new_nonobese - recover_nonobese)
        r_nonobese = min(nonobese_total, r_nonobese + recover_nonobese)

    return pd.DataFrame(rows)


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
    st.title("Obesity and Disease Transmission Dynamics")
    st.markdown(
        """
        <div class="lede">
        Interactive frontend for the ML study: clinical disease-risk analysis,
        obesity-aware epidemic simulation, model comparison, and generated results.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if report:
        cols = st.columns(4)
        dataset = report["dataset"]
        effect = report["obesity_effect_on_disease_risk"]
        epidemic = report["epidemic_simulation"]

        cols[0].metric("Clinical samples", fmt_int(dataset["total_samples"]))
        cols[1].metric("Dataset obesity rate", fmt_pct(dataset["obesity_rate"]))
        cols[2].metric("Heart disease RR", f"{fmt_num(effect['heart_disease_relative_risk'])}x")
        cols[3].metric("Epidemic RR", f"{fmt_num(epidemic['relative_risk_obese_vs_nonobese'])}x")


def render_overview(report: dict) -> None:
    dataset = report["dataset"]
    effect = report["obesity_effect_on_disease_risk"]
    epidemic = report["epidemic_simulation"]

    st.subheader("Study Snapshot")
    cols = st.columns(3)
    cols[0].metric("Stroke rate", fmt_pct(dataset["stroke_rate"]))
    cols[1].metric("Heart disease rate", fmt_pct(dataset["heart_disease_rate"]))
    cols[2].metric("Obese vs non-obese infection risk", f"{fmt_num(epidemic['relative_risk_obese_vs_nonobese'])}x")

    st.markdown(
        """
        <p class="section-note">
        The dashboard reads directly from the generated project artifacts, so updated
        reports and plots will flow into the app without rewriting the UI.
        </p>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1])
    with left:
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

    with right:
        st.subheader("Epidemic Simulation")
        epi_df = pd.DataFrame(
            [
                {"Metric": "Total infected", "Value": fmt_int(epidemic["total_infected"])},
                {"Metric": "Attack rate", "Value": fmt_pct(epidemic["attack_rate"])},
                {"Metric": "Obese attack rate", "Value": fmt_pct(epidemic["obese_attack_rate"])},
                {"Metric": "Non-obese attack rate", "Value": fmt_pct(epidemic["nonobese_attack_rate"])},
                {"Metric": "Peak infected day", "Value": f"Day {epidemic['peak_infected_day']}"},
                {"Metric": "Peak infected count", "Value": fmt_int(epidemic["peak_infected_count"])},
            ]
        )
        st.dataframe(epi_df, use_container_width=True, hide_index=True)

    st.subheader("Generated Visual Evidence")
    image_grid(
        [
            ("Disease rates across BMI groups", "bmi_disease_risk.png"),
            ("Obesity effect summary", "obesity_effect_summary.png"),
            ("Epidemic dynamics", "epidemic_dynamics.png"),
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


def render_epidemic_lab(report: dict) -> None:
    st.subheader("Epidemic What-if Simulator")
    st.markdown(
        """
        <p class="section-note">
        A lightweight two-group SIR simulator for testing how obesity prevalence
        and susceptibility shift epidemic outcomes.
        </p>
        """,
        unsafe_allow_html=True,
    )

    defaults = report["epidemic_simulation"]
    control_cols = st.columns(4)
    population = control_cols[0].slider("Population", 1_000, 50_000, int(defaults["population_size"]), step=1_000)
    days = control_cols[1].slider("Days", 30, 365, int(defaults["simulation_days"]), step=5)
    obesity_rate = control_cols[2].slider("Obesity rate", 0.05, 0.75, float(defaults["obesity_rate"]), step=0.01)
    initial_infected = control_cols[3].slider("Initial infected", 1, 500, int(defaults["initial_infected"]), step=1)

    model_cols = st.columns(4)
    beta = model_cols[0].slider("Infection rate", 0.005, 0.08, float(defaults["base_infection_rate"]), step=0.001)
    gamma = model_cols[1].slider("Recovery rate", 0.02, 0.2, float(defaults["base_recovery_rate"]), step=0.005)
    obesity_multiplier = model_cols[2].slider(
        "Obesity susceptibility multiplier",
        1.0,
        4.0,
        float(defaults["obesity_susceptibility_multiplier"]),
        step=0.1,
    )
    contacts = model_cols[3].slider("Contacts per day", 1.0, 12.0, 5.0, step=0.5)

    history = simulate_epidemic(population, days, obesity_rate, initial_infected, beta, gamma, obesity_multiplier, contacts)
    final = history.iloc[-1]
    peak_row = history.loc[history["infected"].idxmax()]

    obese_total = population * obesity_rate
    nonobese_total = population - obese_total
    obese_attack_rate = final["total_obese_infected"] / obese_total
    nonobese_attack_rate = final["total_nonobese_infected"] / nonobese_total
    relative_risk = obese_attack_rate / nonobese_attack_rate if nonobese_attack_rate else 0

    cols = st.columns(5)
    cols[0].metric("Total infected", fmt_int(final["recovered"] + final["infected"]))
    cols[1].metric("Peak infected", fmt_int(peak_row["infected"]))
    cols[2].metric("Peak day", f"Day {int(peak_row['day'])}")
    cols[3].metric("Obese attack rate", fmt_pct(obese_attack_rate))
    cols[4].metric("Relative risk", f"{fmt_num(relative_risk)}x")

    chart_cols = st.columns([1.2, 0.8])
    with chart_cols[0]:
        st.line_chart(history.set_index("day")[["susceptible", "infected", "recovered"]])
    with chart_cols[1]:
        st.line_chart(history.set_index("day")[["new_infections_obese", "new_infections_nonobese"]])

    attack_df = pd.DataFrame(
        {
            "Group": ["Obese", "Non-obese"],
            "Attack rate": [obese_attack_rate, nonobese_attack_rate],
        }
    ).set_index("Group")
    st.bar_chart(attack_df)


def render_model_performance(report: dict) -> None:
    st.subheader("Model Performance")

    disease_tabs = st.tabs(["Stroke", "Heart disease", "Epidemic forecasting"])
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

    with disease_tabs[2]:
        forecast = report["epidemic_forecasting"]
        target = st.selectbox(
            "Forecast target",
            options=list(forecast.keys()),
            format_func=lambda item: item.replace("_", " ").title(),
        )
        target_report = forecast[target]
        best_key = "best_mae"
        st.metric("Best model", target_report["best_model"], delta=f"MAE {target_report[best_key]:.2f}")
        df = model_table(target_report["all_models"], ["mae", "rmse", "r2"])
        st.dataframe(df.style.format({col: "{:.3f}" for col in df.columns if col != "Model"}), use_container_width=True, hide_index=True)
        st.bar_chart(df.set_index("Model")[["MAE", "RMSE"]])

    st.subheader("Saved Comparison Plots")
    image_grid(
        [
            ("Clinical model comparison", "model_comparison.png"),
            ("Forecasting model comparison", "forecasting_comparison.png"),
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
            "Epidemic simulator",
            "Model performance",
            "Artifacts",
        ],
    )

    st.sidebar.divider()
    st.sidebar.caption("Data source")
    st.sidebar.write(report["dataset"]["name"])
    st.sidebar.caption("Project")
    st.sidebar.write(report["project"])

    if page == "Overview":
        render_overview(report)
    elif page == "Clinical risk scenario":
        render_risk_lab(report)
    elif page == "Epidemic simulator":
        render_epidemic_lab(report)
    elif page == "Model performance":
        render_model_performance(report)
    else:
        render_artifacts(report)


if __name__ == "__main__":
    main()
