"""
Effect of Obesity on Disease Risk using Machine Learning
========================================================

This script builds a focused ML pipeline to study how obesity relates to
individual disease risk using real clinical data from a stroke prediction
dataset with BMI.

Components:
1. Real-data modeling: Predict stroke/heart-disease from BMI and other features
2. ML models: Logistic Regression, Random Forest, Gradient Boosting, XGBoost,
   and Neural Network classifiers
3. Evaluation: Individual risk prediction, obesity effect estimates, and plots
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
# 2. MACHINE LEARNING MODELS FOR INDIVIDUAL DISEASE RISK PREDICTION
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
# 3. VISUALIZATION AND ANALYSIS
# ============================================================================

def create_visualizations(individual_results, feature_cols, df):
    """Create comprehensive visualizations."""
    os.makedirs("results", exist_ok=True)
    
    print("\nCreating visualizations...")
    
    # 1. BMI vs Disease Risk
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
    
    # 2. Model comparison bar charts
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
    
    # 3. Feature importance for stroke model (no target leakage)
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
    
    # 4. Obesity effect summary visualization
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
# 4. SUMMARY REPORT
# ============================================================================

def generate_report(df, individual_results, feature_cols):
    """Generate a comprehensive JSON report."""
    
    obese_stroke = float(df[df["is_obese"] == 1]["stroke"].mean())
    nonobese_stroke = float(df[df["is_obese"] == 0]["stroke"].mean())
    obese_heart = float(df[df["is_obese"] == 1]["heart_disease"].mean())
    nonobese_heart = float(df[df["is_obese"] == 0]["heart_disease"].mean())
    
    # Best models
    best_stroke_model = max(individual_results["stroke"], key=lambda k: individual_results["stroke"][k]["auc"])
    best_heart_model = max(individual_results["heart_disease"], key=lambda k: individual_results["heart_disease"][k]["auc"])
    
    report = {
        "project": "Effect of Obesity on Disease Risk using Machine Learning",
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
        }
    }
    
    for report_path in ["report.json", "results/report.json"]:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")

    print("\nSaved comprehensive report to report.json and results/report.json")
    return report

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)
    print("EFFECT OF OBESITY ON DISEASE RISK USING ML")
    print("="*70)
    
    # 1. Load clinical data
    df = load_and_prepare_clinical_data()
    
    # 2. Train individual risk models
    individual_results, feature_cols = train_individual_risk_models(df)
    
    # 3. Visualizations
    create_visualizations(individual_results, feature_cols, df)
    
    # 4. Generate report
    report = generate_report(df, individual_results, feature_cols)
    
    print("\n" + "="*70)
    print("EXECUTION COMPLETE")
    print("="*70)
    print("\nKey Findings:")
    print(f"1. Obese individuals have {report['obesity_effect_on_disease_risk']['stroke_relative_risk']:.2f}x stroke risk")
    print(f"2. Obese individuals have {report['obesity_effect_on_disease_risk']['heart_disease_relative_risk']:.2f}x heart disease risk")
    print(f"3. Best stroke prediction model: {report['individual_risk_prediction']['stroke']['best_model']} (AUC={report['individual_risk_prediction']['stroke']['best_auc']:.4f})")
    print(f"4. Best heart disease prediction model: {report['individual_risk_prediction']['heart_disease']['best_model']} (AUC={report['individual_risk_prediction']['heart_disease']['best_auc']:.4f})")
    
    print("\nAll results saved to ./results/")
    return report

if __name__ == "__main__":
    main()
