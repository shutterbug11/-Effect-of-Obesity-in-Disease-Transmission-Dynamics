# Effect of Obesity on Disease Risk using Machine Learning

This repository contains a focused machine learning pipeline that investigates
how obesity relates to individual disease risk using clinical data.

## Overview

The project addresses the research question:

**"What is the effect of obesity on disease risk?"**

It uses BMI as an obesity proxy and compares stroke and heart disease outcomes
between obese and non-obese patients.

## Dataset

- **Stroke Prediction Dataset** ([Nnaodeh/Stroke_Prediction_Dataset](https://huggingface.co/datasets/Nnaodeh/Stroke_Prediction_Dataset))
  - 5,110 patient records with 11 clinical features
  - BMI used as obesity proxy (BMI >= 30 = obese)
  - Target variables: stroke and heart_disease

## Methods

### Individual Disease Risk Models

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- Multi-Layer Perceptron

## Key Findings

### Obesity Effect on Individual Disease Risk

| Disease | Obese Rate | Non-obese Rate | Relative Risk |
|---------|------------|----------------|---------------|
| Stroke | 5.10% | 4.73% | **1.08x** |
| Heart Disease | 6.25% | 4.89% | **1.28x** |

### Feature Importance

Top predictive features for stroke prediction:

1. Age
2. Average glucose level
3. Hypertension status
4. BMI / obesity status

## Files

- `obesity_disease_transmission_backup.py` - clinical ML pipeline script
- `report.json` - detailed clinical metrics and model results
- `results/*.png` - generated visualizations
- `app.py` - Streamlit dashboard

## Reproduction

```bash
pip install -r requirements.txt
python obesity_disease_transmission_backup.py
```

## Streamlit Frontend

Run the interactive dashboard from the project root:

```bash
streamlit run app.py
```

The app reads `report.json` and the generated clinical figures to provide:

- A project overview with the main obesity-risk findings
- A clinical risk scenario tool based on aggregate study rates
- Model-performance tables and comparison charts
- A raw artifact browser for the generated report and feature set

## Citation

- Stroke Prediction Dataset: [Nnaodeh/Stroke_Prediction_Dataset](https://huggingface.co/datasets/Nnaodeh/Stroke_Prediction_Dataset)
