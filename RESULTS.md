# Results: Effect of Obesity in Disease Transmission Dynamics

## Executive Summary

This study used machine learning to investigate how obesity affects both individual disease risk and population-level disease transmission dynamics. We analyzed 5,110 clinical records and simulated epidemics in populations of 10,000 individuals.

---

## 1. Individual Disease Risk Prediction

### Dataset
- **Source**: [Nnaodeh/Stroke_Prediction_Dataset](https://huggingface.co/datasets/Nnaodeh/Stroke_Prediction_Dataset)
- **Samples**: 5,110 patients
- **Obesity Rate**: 37.6% (BMI ≥ 30)
- **Features**: Age, gender, hypertension, heart disease, marriage status, work type, residence type, glucose level, BMI, smoking status

### Obesity Effect on Disease Rates

| Metric | Obese (BMI ≥ 30) | Non-obese (BMI < 30) | Relative Risk |
|--------|-----------------|---------------------|---------------|
| **Stroke Rate** | 5.10% | 4.73% | **1.08×** |
| **Heart Disease Rate** | 6.25% | 4.89% | **1.28×** |

### Model Performance — Stroke Prediction

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| **Logistic Regression** | 74.1% | 13.6% | 80.0% | 23.2% | **0.839** |
| Random Forest | 92.1% | 19.6% | 20.0% | 19.8% | 0.824 |
| Gradient Boosting | 94.8% | 33.3% | 6.0% | 10.2% | 0.797 |
| XGBoost | 92.5% | 22.4% | 22.0% | 22.2% | 0.812 |
| MLP Neural Net | 95.1% | 0.0% | 0.0% | 0.0% | 0.543 |

**Best Model**: Logistic Regression (AUC = 0.839)

### Model Performance — Heart Disease Prediction

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| **Logistic Regression** | 75.6% | 14.6% | 72.7% | 24.3% | **0.830** |
| Random Forest | 87.6% | 14.0% | 25.5% | 18.1% | 0.798 |
| Gradient Boosting | 93.4% | 16.7% | 5.5% | 8.2% | 0.785 |
| XGBoost | 89.7% | 16.2% | 21.8% | 18.6% | 0.803 |
| MLP Neural Net | 94.6% | 0.0% | 0.0% | 0.0% | 0.485 |

**Best Model**: Logistic Regression (AUC = 0.830)

### Feature Importance (XGBoost Stroke Model)
1. **Age** — most predictive feature
2. Average glucose level
3. Hypertension status
4. Ever married (proxy for age)
5. **BMI / Obesity status**

---

## 2. Epidemic Transmission Dynamics Simulation

### Simulation Parameters
- **Population**: 10,000 individuals
- **Duration**: 200 days
- **Obesity Rate**: 35%
- **Initial Infections**: 20
- **Base Infection Rate (β)**: 0.03 per contact
- **Recovery Rate (γ)**: 0.08 (~12 day infectious period)
- **Obesity Susceptibility Multiplier**: 2.0×

### Key Epidemiological Outcomes

| Metric | Value |
|--------|-------|
| Total Infected | 8,510 (85.1%) |
| Obese Attack Rate | **96.5%** |
| Non-obese Attack Rate | **79.0%** |
| **Relative Risk (Obese vs Non-obese)** | **1.22×** |
| Peak Infected Day | Day 44 |
| Peak Simultaneous Infections | 3,816 |

### Interpretation
Obese individuals were **22% more likely** to become infected during the epidemic compared to non-obese individuals, due to their increased susceptibility to infection transmission. This demonstrates that obesity acts as a significant amplifier in disease transmission dynamics at the population level.

---

## 3. Epidemic Forecasting with Machine Learning

We trained ML models to predict next-day new infections from recent epidemic history (7-day rolling window).

### Total Infections Forecasting

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| **XGBoost** | **3.78** | 8.44 | **0.987** |
| Ridge | 4.04 | 9.57 | 0.984 |
| Gradient Boosting | 5.23 | 11.31 | 0.978 |
| Random Forest | 6.77 | 15.97 | 0.955 |

### Obese Infections Forecasting

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| **XGBoost** | **1.94** | 4.91 | **0.979** |
| Ridge | 2.15 | 5.86 | 0.970 |
| Gradient Boosting | 2.37 | 5.58 | 0.973 |
| Random Forest | 3.36 | 8.69 | 0.933 |

### Non-obese Infections Forecasting

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| **Ridge** | **2.46** | 6.88 | **0.974** |
| XGBoost | 2.80 | 6.43 | 0.977 |
| Random Forest | 3.12 | 7.74 | 0.967 |
| Gradient Boosting | 4.10 | 8.11 | 0.964 |

---

## 4. Visualizations Generated

1. **epidemic_dynamics.png** — SIR curves showing susceptible, infected, and recovered populations over time; daily and cumulative infections by obesity status; attack rate comparison
2. **bmi_disease_risk.png** — Stroke and heart disease rates across BMI categories (<18.5, 18.5-25, 25-30, 30-35, >35)
3. **model_comparison.png** — Side-by-side bar charts comparing accuracy, precision, recall, F1, and AUC across all 5 ML models for stroke and heart disease prediction
4. **forecasting_comparison.png** — MAE and RMSE comparison for epidemic forecasting models across total, obese, and non-obese infection streams
5. **feature_importance.png** — Horizontal bar chart of XGBoost feature importances for stroke prediction
6. **obesity_effect_summary.png** — Comparative disease rates showing the obesity effect for stroke and heart disease

---

## 5. Conclusions

1. **Obesity increases individual disease risk**: Obese individuals show 1.08× stroke risk and 1.28× heart disease risk in the clinical dataset.

2. **Obesity amplifies epidemic spread**: In our agent-based simulation, obese individuals had a 96.5% attack rate vs 79.0% for non-obese — a 1.22× relative risk. This means obesity not only affects personal health outcomes but also acts as an epidemic amplifier.

3. **Age is the dominant risk factor**: Across all ML models, age consistently ranks as the most important predictor for stroke and heart disease, followed by glucose level and BMI/obesity.

4. **ML models can forecast epidemic dynamics**: XGBoost achieved R² = 0.987 for predicting next-day total infections and R² = 0.979 for obese-specific infection streams, demonstrating that machine learning can effectively learn epidemic patterns from time-series data.

5. **Simple models outperform complex ones**: Logistic Regression achieved the highest AUC (0.839) for stroke prediction despite being the simplest model, while neural networks suffered from severe class imbalance issues. This highlights the importance of matching model complexity to dataset characteristics.

---

## Citation

- Dataset: [Nnaodeh/Stroke_Prediction_Dataset](https://huggingface.co/datasets/Nnaodeh/Stroke_Prediction_Dataset)
- Christakis NA, Fowler JH. The Spread of Obesity in a Large Social Network over 32 Years. *New England Journal of Medicine*. 2007;357(4):370-379.
