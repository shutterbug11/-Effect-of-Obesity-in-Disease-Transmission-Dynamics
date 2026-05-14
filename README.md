# Effect of Obesity in Disease Transmission Dynamics using Machine Learning

This repository contains a comprehensive machine learning pipeline that investigates how obesity affects both individual disease risk and population-level disease transmission dynamics.

## Overview

The project addresses the research question: **"What is the effect of obesity in disease transmission dynamics?"** using machine learning approaches across two complementary domains:

1. **Individual Risk Prediction**: Predicting stroke and heart disease from clinical features including BMI/obesity status
2. **Population Transmission Dynamics**: Simulating and forecasting epidemic spread with obesity as a comorbidity risk modifier

## Datasets

- **Stroke Prediction Dataset** ([Nnaodeh/Stroke_Prediction_Dataset](https://huggingface.co/datasets/Nnaodeh/Stroke_Prediction_Dataset))
  - 5,110 patient records with 11 clinical features
  - BMI used as obesity proxy (BMI ≥ 30 = obese)
  - Target variables: stroke and heart_disease

## Methods

### Individual Disease Risk Models
- Logistic Regression (best for stroke: AUC = 0.839)
- Random Forest
- Gradient Boosting
- XGBoost
- Multi-Layer Perceptron (Neural Network)

### Epidemic Simulation
- Agent-based SIR model (Susceptible-Infected-Recovered)
- Population: 10,000 individuals
- Obesity susceptibility multiplier: 2.0x
- Base infection rate: 0.03 per contact
- Recovery rate: 0.08 (~12 day infectious period)

### Epidemic Forecasting Models
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor (best: MAE = 3.78 for total infections)

## Key Findings

### Obesity Effect on Individual Disease Risk
| Disease | Obese Rate | Non-obese Rate | Relative Risk |
|---------|-----------|----------------|---------------|
| Stroke | 5.10% | 4.73% | **1.08x** |
| Heart Disease | 6.25% | 4.89% | **1.28x** |

### Obesity Effect on Epidemic Transmission
- **Obese attack rate**: 96.49%
- **Non-obese attack rate**: 78.97%
- **Relative risk**: **1.22x**
- Peak infected day: Day 44 (3,816 simultaneous infections)

### Feature Importance (Stroke Prediction)
Top predictive features:
1. **Age** (most important)
2. Average glucose level
3. BMI / Obesity status

## Files

- `obesity_disease_transmission.py` — Complete pipeline script
- `report.json` — Detailed metrics and results
- `results/*.png` — Visualizations

## Reproduction

```bash
pip install -r requirements.txt
python obesity_disease_transmission.py
```

## Streamlit Frontend

Run the interactive dashboard from the project root:

```bash
streamlit run app.py
```

The app reads `report.json` and the generated figures in `results/` to provide:

- A project overview with the main obesity-risk and epidemic findings
- A clinical risk scenario tool based on aggregate study rates
- An interactive obesity-aware epidemic what-if simulator
- Model-performance tables and comparison charts
- A raw artifact browser for the generated report and feature set

## Citation

- Stroke Prediction Dataset: [Nnaodeh/Stroke_Prediction_Dataset](https://huggingface.co/datasets/Nnaodeh/Stroke_Prediction_Dataset)
- Christakis & Fowler (2007): "The Spread of Obesity in a Large Social Network over 32 Years"

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = 'Agniv1/obesity-disease-transmission-ml'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
```

For non-causal architectures, replace `AutoModelForCausalLM` with the appropriate `AutoModel` class.
