"""
Effect of Obesity in Disease Transmission Dynamics using Machine Learning
============================================================================

This script builds a comprehensive ML pipeline to study how obesity affects
disease transmission dynamics. It combines real clinical data (stroke prediction
dataset with BMI) with simulated epidemic dynamics where obesity acts as a
comorbidity risk modifier.

Components:
1. Real-data modeling: Predict stroke/heart-disease from BMI and other features
2. Epidemic simulation: SIR-like model with obesity-dependent susceptibility
3. ML models: Random Forest, XGBoost, Neural Network for both tasks
4. Evaluation: Individual risk prediction + population transmission forecasting
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
import xgboost as xgb
from datasets import load_dataset
import json
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Epidemic simulation parameters
POPULATION_SIZE = 10000
SIMULATION_DAYS = 200
BASE_RECOVERY_RATE = 0.08  # ~12 day average infectious period
BASE_INFECTION_RATE = 0.03  # Lower per-contact transmission probability
OBESITY_SUSCEPTIBILITY_MULTIPLIER = 2.0  # Obese individuals 2x more susceptible
INITIAL_INFECTED = 20

# ============================================================================
# 1. LOAD AND PREPARE REAL CLINICAL DATA
# ============================================================================

def load_and_prepare_clinical_data():
    """Load stroke prediction dataset with BMI as obesity proxy."""
    print("Loading clinical data (Stroke Prediction Dataset with BMI)...")
    ds = load_dataset("Nnaodeh/Stroke_Prediction_Dataset", split="train")
    df = ds.to_pandas()
    print(f"Original shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values per column:\n{df.isnull().sum()}")
    
    # Drop ID column
    df = df.drop(columns=["id"])
    
    # Handle missing BMI values
    df["bmi"] = df["bmi"].replace("N/A", np.nan)
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    
    # Create obesity categories based on WHO BMI standards
    def bmi_category(bmi):
        if pd.isna(bmi):
            return "unknown"
        if bmi < 18.5:
            return "underweight"
        elif bmi < 25:
            return "normal"
        elif bmi < 30:
            return "overweight"
        else:
            return "obese"
    
    df["bmi_category"] = df["bmi"].apply(bmi_category)
    
    # Create binary obesity indicator
    df["is_obese"] = (df["bmi"] >= 30).astype(int)
    
    # Encode categorical variables
    le_gender = LabelEncoder()
    df["gender_enc"] = le_gender.fit_transform(df["gender"])
    
    le_married = LabelEncoder()
    df["ever_married_enc"] = le_married.fit_transform(df["ever_married"])
    
    le_work = LabelEncoder()
    df["work_type_enc"] = le_work.fit_transform(df["work_type"])
    
    le_residence = LabelEncoder()
    df["Residence_type_enc"] = le_residence.fit_transform(df["Residence_type"])
    
    le_smoking = LabelEncoder()
    df["smoking_status_enc"] = le_smoking.fit_transform(df["smoking_status"])
    
    # Impute missing BMI with median
    imputer = SimpleImputer(strategy="median")
    df[["bmi"]] = imputer.fit_transform(df[["bmi"]])
    
    print(f"\nAfter preprocessing shape: {df.shape}")
    print(f"\nBMI distribution:")
    print(df["bmi_category"].value_counts())
    print(f"\nStroke rate by obesity status:")
    print(df.groupby("is_obese")["stroke"].mean())
    print(f"\nHeart disease rate by obesity status:")
    print(df.groupby("is_obese")["heart_disease"].mean())
    
    return df

# ============================================================================
# 2. EPIDEMIC TRANSMISSION SIMULATION WITH OBESITY FACTOR
# ============================================================================

def simulate_epidemic_with_obesity(pop_size=POPULATION_SIZE, days=SIMULATION_DAYS,
                                   base_beta=BASE_INFECTION_RATE,
                                   gamma=BASE_RECOVERY_RATE,
                                   obesity_multiplier=OBESITY_SUSCEPTIBILITY_MULTIPLIER,
                                   initial_infected=INITIAL_INFECTED,
                                   obesity_rate=0.35):
    """
    Simulate SIR model where obesity increases susceptibility to infection.
    
    Each individual has:
    - S: Susceptible
    - I: Infected
    - R: Recovered
    - obesity_status: 0/1 affecting their infection probability
    
    Transmission rate for contact between i (infected) and j (susceptible):
        beta_eff = base_beta * (1 if j not obese else obesity_multiplier)
    """
    print(f"\nSimulating epidemic with obesity effect...")
    print(f"Population: {pop_size}, Days: {days}, Obesity rate: {obesity_rate}")
    print(f"Base beta: {base_beta}, Obesity multiplier: {obesity_multiplier}")
    
    # Initialize population
    np.random.seed(RANDOM_STATE)
    population = pd.DataFrame({
        "id": range(pop_size),
        "obese": (np.random.rand(pop_size) < obesity_rate).astype(int),
        "state": "S",
        "infection_day": -1,
        "recovery_day": -1,
    })
    
    # Infect initial individuals (random, proportional to obesity for realism)
    infection_probs = population["obese"] * obesity_multiplier + (1 - population["obese"])
    infection_probs = infection_probs / infection_probs.sum()
    initial_indices = np.random.choice(pop_size, size=initial_infected,
                                      replace=False, p=infection_probs)
    population.loc[initial_indices, "state"] = "I"
    population.loc[initial_indices, "infection_day"] = 0
    
    # Track daily statistics
    history = {
        "day": [],
        "S": [],
        "I": [],
        "R": [],
        "new_infections": [],
        "new_infections_obese": [],
        "new_infections_nonobese": [],
        "total_obese_infected": [],
        "total_nonobese_infected": [],
    }
    
    for day in range(days):
        infected = population[population["state"] == "I"]
        susceptible = population[population["state"] == "S"]
        
        new_infections = 0
        new_inf_obese = 0
        new_inf_nonobese = 0
        
        # For each susceptible person, chance of infection from any infected
        if len(infected) > 0 and len(susceptible) > 0:
            # Simplified: each susceptible meets random infected with some probability
            # meeting_rate determines how many contacts per day
            meeting_rate = 5.0 / pop_size
            
            for idx in susceptible.index:
                obese = population.loc[idx, "obese"]
                beta_eff = base_beta * (obesity_multiplier if obese else 1.0)
                
                # Probability of at least one transmission
                n_contacts = np.random.poisson(meeting_rate * len(infected))
                if n_contacts > 0:
                    p_no_transmission = (1 - beta_eff) ** n_contacts
                    if np.random.rand() > p_no_transmission:
                        population.loc[idx, "state"] = "I"
                        population.loc[idx, "infection_day"] = day
                        new_infections += 1
                        if obese:
                            new_inf_obese += 1
                        else:
                            new_inf_nonobese += 1
        
        # Recovery: deterministic threshold based on average infectious period
        recover_threshold = max(1, int(round(1.0 / gamma)))
        newly_recovered_mask = (
            (population["state"] == "I") &
            (day - population["infection_day"] >= recover_threshold)
        )
        newly_recovered_idx = population[newly_recovered_mask].index
        population.loc[newly_recovered_idx, "state"] = "R"
        population.loc[newly_recovered_idx, "recovery_day"] = day
        
        # Record state
        s_count = (population["state"] == "S").sum()
        i_count = (population["state"] == "I").sum()
        r_count = (population["state"] == "R").sum()
        
        obese_infected = ((population["state"].isin(["I", "R"])) & (population["obese"] == 1)).sum()
        nonobese_infected = ((population["state"].isin(["I", "R"])) & (population["obese"] == 0)).sum()
        
        history["day"].append(day)
        history["S"].append(s_count)
        history["I"].append(i_count)
        history["R"].append(r_count)
        history["new_infections"].append(new_infections)
        history["new_infections_obese"].append(new_inf_obese)
        history["new_infections_nonobese"].append(new_inf_nonobese)
        history["total_obese_infected"].append(obese_infected)
        history["total_nonobese_infected"].append(nonobese_infected)
        
        if day % 20 == 0:
            print(f"Day {day}: S={s_count}, I={i_count}, R={r_count}, "
                  f"New infections={new_infections}")
    
    history_df = pd.DataFrame(history)
    
    # Calculate key epidemiological metrics
    total_infected = pop_size - history_df["S"].iloc[-1]
    attack_rate = total_infected / pop_size
    obese_attack_rate = history_df["total_obese_infected"].iloc[-1] / (pop_size * obesity_rate)
    nonobese_attack_rate = history_df["total_nonobese_infected"].iloc[-1] / (pop_size * (1 - obesity_rate))
    
    print(f"\nEpidemic Summary:")
    print(f"Total infected: {total_infected} ({attack_rate:.2%})")
    print(f"Obese attack rate: {obese_attack_rate:.2%}")
    print(f"Non-obese attack rate: {nonobese_attack_rate:.2%}")
    print(f"Relative risk (obese vs non-obese): {obese_attack_rate / nonobese_attack_rate:.2f}x")
    
    return population, history_df

# ============================================================================
# 3. MACHINE LEARNING MODELS FOR INDIVIDUAL DISEASE RISK PREDICTION
# ============================================================================

def train_individual_risk_models(df):
    """Train ML models to predict stroke/heart disease from features including BMI."""
    print("\n" + "="*60)
    print("TRAINING INDIVIDUAL DISEASE RISK MODELS")
    print("="*60)
    
    feature_cols = ["gender_enc", "age", "hypertension",
                    "ever_married_enc", "work_type_enc", "Residence_type_enc",
                    "avg_glucose_level", "bmi", "smoking_status_enc", "is_obese"]
    
    X_stroke = df[feature_cols].copy()
    X_heart = df[feature_cols].copy()
    y_stroke = df["stroke"].values
    y_heart = df["heart_disease"].values
    
    results = {}
    
    for target_name, X, y in [("stroke", X_stroke, y_stroke),
                               ("heart_disease", X_heart, y_heart)]:
        print(f"\n--- Predicting {target_name.upper()} ---")
        print(f"Class distribution: {np.bincount(y)}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced",
                                                       random_state=RANDOM_STATE),
            "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10,
                                                    class_weight="balanced",
                                                    random_state=RANDOM_STATE, n_jobs=-1),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                                            random_state=RANDOM_STATE),
            "XGBoost": xgb.XGBClassifier(n_estimators=200, max_depth=5,
                                          learning_rate=0.1, subsample=0.8,
                                          colsample_bytree=0.8, scale_pos_weight=10,
                                          random_state=RANDOM_STATE, eval_metric="logloss"),
            "MLP_NeuralNet": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                            early_stopping=True, random_state=RANDOM_STATE)
        }
        
        target_results = {}
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            
            if model_name in ["LogisticRegression", "MLP_NeuralNet"]:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_prob)
            
            print(f"  Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
            
            # Feature importance for tree-based models
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
                print(f"  Top 3 important features: {dict(feat_imp.head(3))}")
            elif hasattr(model, "coef_"):
                coefs = np.abs(model.coef_[0])
                feat_imp = pd.Series(coefs, index=feature_cols).sort_values(ascending=False)
                print(f"  Top 3 important features: {dict(feat_imp.head(3))}")
            
            target_results[model_name] = {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "auc": float(auc),
                "predictions": y_pred.tolist(),
                "probabilities": y_prob.tolist(),
                "y_test": y_test.tolist()
            }
        
        results[target_name] = target_results
    
    return results, feature_cols

# ============================================================================
# 4. ML MODELS FOR EPIDEMIC DYNAMICS FORECASTING
# ============================================================================

def prepare_epidemic_ml_data(history_df, window_size=7):
    """
    Create supervised learning dataset from epidemic time series.
    Predict next-day infections from recent history.
    """
    print("\nPreparing epidemic forecasting dataset...")
    
    features = []
    targets_total = []
    targets_obese = []
    targets_nonobese = []
    
    for i in range(window_size, len(history_df)):
        window = history_df.iloc[i-window_size:i]
        feat = [
            window["S"].mean(),
            window["I"].mean(),
            window["R"].mean(),
            window["new_infections"].mean(),
            window["new_infections_obese"].mean(),
            window["new_infections_nonobese"].mean(),
            window["S"].std(),
            window["I"].std(),
            window["new_infections"].max(),
            window["new_infections"].sum(),
            i,  # day number
        ]
        features.append(feat)
        targets_total.append(history_df.iloc[i]["new_infections"])
        targets_obese.append(history_df.iloc[i]["new_infections_obese"])
        targets_nonobese.append(history_df.iloc[i]["new_infections_nonobese"])
    
    X = np.array(features)
    y_total = np.array(targets_total)
    y_obese = np.array(targets_obese)
    y_nonobese = np.array(targets_nonobese)
    
    return X, y_total, y_obese, y_nonobese

def train_epidemic_forecast_models(X, y_total, y_obese, y_nonobese):
    """Train models to forecast daily new infections."""
    print("\n" + "="*60)
    print("TRAINING EPIDEMIC FORECASTING MODELS")
    print("="*60)
    
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    results = {}
    
    for target_name, y in [("total_infections", y_total),
                             ("obese_infections", y_obese),
                             ("nonobese_infections", y_nonobese)]:
        print(f"\n--- Forecasting {target_name.upper()} ---")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
            "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=10,
                                                  random_state=RANDOM_STATE, n_jobs=-1),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                                           random_state=RANDOM_STATE),
            "XGBoost": xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1,
                                         subsample=0.8, random_state=RANDOM_STATE)
        }
        
        target_results = {}
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            
            if model_name == "Ridge":
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            print(f"  MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f}")
            
            target_results[model_name] = {
                "mae": float(mae),
                "rmse": float(rmse),
                "r2": float(r2),
                "predictions": y_pred.tolist(),
                "actual": y_test.tolist()
            }
        
        results[target_name] = target_results
    
    return results

# ============================================================================
# 5. VISUALIZATION AND ANALYSIS
# ============================================================================

def create_visualizations(history_df, individual_results, epidemic_results,
                          population, feature_cols, df):
    """Create comprehensive visualizations."""
    os.makedirs("results", exist_ok=True)
    
    print("\nCreating visualizations...")
    
    # 1. Epidemic curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    ax.plot(history_df["day"], history_df["S"], label="Susceptible", color="green")
    ax.plot(history_df["day"], history_df["I"], label="Infected", color="red")
    ax.plot(history_df["day"], history_df["R"], label="Recovered", color="blue")
    ax.set_xlabel("Day")
    ax.set_ylabel("Population")
    ax.set_title("SIR Epidemic Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(history_df["day"], history_df["new_infections_obese"], label="Obese", color="orange")
    ax.plot(history_df["day"], history_df["new_infections_nonobese"], label="Non-obese", color="teal")
    ax.set_xlabel("Day")
    ax.set_ylabel("New Infections")
    ax.set_title("Daily New Infections by Obesity Status")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(history_df["day"], history_df["total_obese_infected"], label="Cumulative Obese Infected", color="orange")
    ax.plot(history_df["day"], history_df["total_nonobese_infected"], label="Cumulative Non-obese Infected", color="teal")
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative Infected")
    ax.set_title("Cumulative Infections by Obesity Status")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    obese_rate = history_df["total_obese_infected"] / (POPULATION_SIZE * 0.35)
    nonobese_rate = history_df["total_nonobese_infected"] / (POPULATION_SIZE * 0.65)
    ax.plot(history_df["day"], obese_rate, label="Obese Attack Rate", color="orange")
    ax.plot(history_df["day"], nonobese_rate, label="Non-obese Attack Rate", color="teal")
    ax.set_xlabel("Day")
    ax.set_ylabel("Attack Rate")
    ax.set_title("Attack Rate Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/epidemic_dynamics.png", dpi=150)
    plt.close()
    print("  Saved: results/epidemic_dynamics.png")
    
    # 2. BMI vs Disease Risk
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    bmi_bins = pd.cut(df["bmi"], bins=[0, 18.5, 25, 30, 35, 100], labels=["<18.5", "18.5-25", "25-30", "30-35", ">35"])
    stroke_by_bmi = df.groupby(bmi_bins)["stroke"].mean()
    heart_by_bmi = df.groupby(bmi_bins)["heart_disease"].mean()
    
    ax = axes[0]
    stroke_by_bmi.plot(kind="bar", ax=ax, color="crimson", alpha=0.7)
    ax.set_title("Stroke Rate by BMI Category")
    ax.set_ylabel("Stroke Rate")
    ax.set_xlabel("BMI Category")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    heart_by_bmi.plot(kind="bar", ax=ax, color="navy", alpha=0.7)
    ax.set_title("Heart Disease Rate by BMI Category")
    ax.set_ylabel("Heart Disease Rate")
    ax.set_xlabel("BMI Category")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/bmi_disease_risk.png", dpi=150)
    plt.close()
    print("  Saved: results/bmi_disease_risk.png")
    
    # 3. Model comparison bar charts
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, target in enumerate(["stroke", "heart_disease"]):
        ax = axes[idx]
        models = list(individual_results[target].keys())
        metrics = ["accuracy", "precision", "recall", "f1", "auc"]
        x = np.arange(len(models))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            values = [individual_results[target][m][metric] for m in models]
            ax.bar(x + i * width, values, width, label=metric)
        
        ax.set_ylabel("Score")
        ax.set_title(f"Model Performance: {target.upper()} Prediction")
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig("results/model_comparison.png", dpi=150)
    plt.close()
    print("  Saved: results/model_comparison.png")
    
    # 4. Forecasting comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for idx, target in enumerate(["total_infections", "obese_infections", "nonobese_infections"]):
        ax = axes[idx]
        models = list(epidemic_results[target].keys())
        x = np.arange(len(models))
        width = 0.25
        
        mae_vals = [epidemic_results[target][m]["mae"] for m in models]
        rmse_vals = [epidemic_results[target][m]["rmse"] for m in models]
        
        ax.bar(x - width/2, mae_vals, width, label="MAE", color="coral")
        ax.bar(x + width/2, rmse_vals, width, label="RMSE", color="skyblue")
        
        ax.set_ylabel("Error")
        ax.set_title(f"Forecasting Error: {target.replace('_', ' ').title()}")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig("results/forecasting_comparison.png", dpi=150)
    plt.close()
    print("  Saved: results/forecasting_comparison.png")
    
    # 5. Feature importance for stroke model (no target leakage)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    feat_cols_no_leak = ["gender_enc", "age", "hypertension",
                         "ever_married_enc", "work_type_enc", "Residence_type_enc",
                         "avg_glucose_level", "bmi", "smoking_status_enc", "is_obese"]
    X_plot = df[feat_cols_no_leak]
    y_plot = df["stroke"].values
    
    model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                               scale_pos_weight=10, random_state=RANDOM_STATE, eval_metric="logloss")
    model.fit(X_plot, y_plot)
    
    importances = pd.Series(model.feature_importances_, index=feat_cols_no_leak).sort_values()
    importances.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Feature Importance: Stroke Prediction (XGBoost)")
    ax.set_xlabel("Importance")
    ax.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    plt.savefig("results/feature_importance.png", dpi=150)
    plt.close()
    print("  Saved: results/feature_importance.png")
    
    # 6. Obesity effect summary visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    
    obese_stroke = df[df["is_obese"] == 1]["stroke"].mean()
    nonobese_stroke = df[df["is_obese"] == 0]["stroke"].mean()
    obese_heart = df[df["is_obese"] == 1]["heart_disease"].mean()
    nonobese_heart = df[df["is_obese"] == 0]["heart_disease"].mean()
    
    categories = ["Stroke (Obese)", "Stroke (Non-obese)", "Heart Disease (Obese)", "Heart Disease (Non-obese)"]
    values = [obese_stroke, nonobese_stroke, obese_heart, nonobese_heart]
    colors = ["darkred", "lightcoral", "darkblue", "lightblue"]
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8)
    ax.set_ylabel("Disease Rate")
    ax.set_title("Disease Rates by Obesity Status")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    
    plt.tight_layout()
    plt.savefig("results/obesity_effect_summary.png", dpi=150)
    plt.close()
    print("  Saved: results/obesity_effect_summary.png")
    
    return True

# ============================================================================
# 6. SUMMARY REPORT
# ============================================================================

def generate_report(df, population, history_df, individual_results,
                    epidemic_results, feature_cols):
    """Generate a comprehensive JSON report."""
    
    obese_stroke = float(df[df["is_obese"] == 1]["stroke"].mean())
    nonobese_stroke = float(df[df["is_obese"] == 0]["stroke"].mean())
    obese_heart = float(df[df["is_obese"] == 1]["heart_disease"].mean())
    nonobese_heart = float(df[df["is_obese"] == 0]["heart_disease"].mean())
    
    total_infected = int(POPULATION_SIZE - history_df["S"].iloc[-1])
    attack_rate = float(total_infected / POPULATION_SIZE)
    obese_attack = float(history_df["total_obese_infected"].iloc[-1] / (POPULATION_SIZE * 0.35))
    nonobese_attack = float(history_df["total_nonobese_infected"].iloc[-1] / (POPULATION_SIZE * 0.65))
    
    # Best models
    best_stroke_model = max(individual_results["stroke"], key=lambda k: individual_results["stroke"][k]["auc"])
    best_heart_model = max(individual_results["heart_disease"], key=lambda k: individual_results["heart_disease"][k]["auc"])
    
    report = {
        "project": "Effect of Obesity in Disease Transmission Dynamics using Machine Learning",
        "dataset": {
            "name": "Stroke Prediction Dataset (Nnaodeh/Stroke_Prediction_Dataset)",
            "total_samples": len(df),
            "features": feature_cols,
            "obesity_rate": float(df["is_obese"].mean()),
            "stroke_rate": float(df["stroke"].mean()),
            "heart_disease_rate": float(df["heart_disease"].mean())
        },
        "obesity_effect_on_disease_risk": {
            "stroke_rate_obese": obese_stroke,
            "stroke_rate_nonobese": nonobese_stroke,
            "stroke_relative_risk": float(obese_stroke / nonobese_stroke) if nonobese_stroke > 0 else None,
            "heart_disease_rate_obese": obese_heart,
            "heart_disease_rate_nonobese": nonobese_heart,
            "heart_disease_relative_risk": float(obese_heart / nonobese_heart) if nonobese_heart > 0 else None
        },
        "epidemic_simulation": {
            "population_size": POPULATION_SIZE,
            "simulation_days": SIMULATION_DAYS,
            "base_infection_rate": BASE_INFECTION_RATE,
            "base_recovery_rate": BASE_RECOVERY_RATE,
            "obesity_susceptibility_multiplier": OBESITY_SUSCEPTIBILITY_MULTIPLIER,
            "initial_infected": INITIAL_INFECTED,
            "obesity_rate": 0.35,
            "total_infected": total_infected,
            "attack_rate": attack_rate,
            "obese_attack_rate": obese_attack,
            "nonobese_attack_rate": nonobese_attack,
            "relative_risk_obese_vs_nonobese": float(obese_attack / nonobese_attack) if nonobese_attack > 0 else None,
            "peak_infected_day": int(history_df.loc[history_df["I"].idxmax(), "day"]),
            "peak_infected_count": int(history_df["I"].max())
        },
        "individual_risk_prediction": {
            "stroke": {
                "best_model": best_stroke_model,
                "best_auc": individual_results["stroke"][best_stroke_model]["auc"],
                "all_models": {k: {m: v[m] for m in ["accuracy", "precision", "recall", "f1", "auc"]}
                               for k, v in individual_results["stroke"].items()}
            },
            "heart_disease": {
                "best_model": best_heart_model,
                "best_auc": individual_results["heart_disease"][best_heart_model]["auc"],
                "all_models": {k: {m: v[m] for m in ["accuracy", "precision", "recall", "f1", "auc"]}
                               for k, v in individual_results["heart_disease"].items()}
            }
        },
        "epidemic_forecasting": {
            target: {
                "best_model": min(epidemic_results[target], key=lambda k: epidemic_results[target][k]["mae"]),
                "best_mae": min(v["mae"] for v in epidemic_results[target].values()),
                "all_models": {k: {m: v[m] for m in ["mae", "rmse", "r2"]}
                               for k, v in epidemic_results[target].items()}
            }
            for target in epidemic_results
        }
    }
    
    with open("results/report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\nSaved comprehensive report to results/report.json")
    return report

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)
    print("EFFECT OF OBESITY IN DISEASE TRANSMISSION DYNAMICS USING ML")
    print("="*70)
    
    # 1. Load clinical data
    df = load_and_prepare_clinical_data()
    
    # 2. Simulate epidemic with obesity effect
    population, history_df = simulate_epidemic_with_obesity()
    
    # 3. Train individual risk models
    individual_results, feature_cols = train_individual_risk_models(df)
    
    # 4. Train epidemic forecasting models
    X_epi, y_total, y_obese, y_nonobese = prepare_epidemic_ml_data(history_df)
    epidemic_results = train_epidemic_forecast_models(X_epi, y_total, y_obese, y_nonobese)
    
    # 5. Visualizations
    create_visualizations(history_df, individual_results, epidemic_results,
                          population, feature_cols, df)
    
    # 6. Generate report
    report = generate_report(df, population, history_df, individual_results,
                            epidemic_results, feature_cols)
    
    print("\n" + "="*70)
    print("EXECUTION COMPLETE")
    print("="*70)
    print("\nKey Findings:")
    print(f"1. Obese individuals have {report['obesity_effect_on_disease_risk']['stroke_relative_risk']:.2f}x stroke risk")
    print(f"2. Obese individuals have {report['obesity_effect_on_disease_risk']['heart_disease_relative_risk']:.2f}x heart disease risk")
    print(f"3. In epidemic simulation, obese attack rate = {report['epidemic_simulation']['obese_attack_rate']:.2%}")
    print(f"4. Non-obese attack rate = {report['epidemic_simulation']['nonobese_attack_rate']:.2%}")
    print(f"5. Epidemic relative risk (obese vs non-obese) = {report['epidemic_simulation']['relative_risk_obese_vs_nonobese']:.2f}x")
    print(f"6. Best stroke prediction model: {report['individual_risk_prediction']['stroke']['best_model']} (AUC={report['individual_risk_prediction']['stroke']['best_auc']:.4f})")
    print(f"7. Best epidemic forecaster: {report['epidemic_forecasting']['total_infections']['best_model']} (MAE={report['epidemic_forecasting']['total_infections']['best_mae']:.2f})")
    
    print("\nAll results saved to ./results/")
    return report

if __name__ == "__main__":
    main()
