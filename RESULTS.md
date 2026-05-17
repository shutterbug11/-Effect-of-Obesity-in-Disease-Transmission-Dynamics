# Results: Effect of Obesity on Disease Risk

## Executive Summary

This study used machine learning to investigate how obesity relates to
individual disease risk using clinical data.

## 1. Individual Disease Risk Prediction

### Dataset

- **Source**: [Nnaodeh/Stroke_Prediction_Dataset](https://huggingface.co/datasets/Nnaodeh/Stroke_Prediction_Dataset)
- **Samples**: 5,110 patients
- **Obesity Rate**: 37.6% (BMI >= 30)
- **Features**: Age, gender, hypertension, heart disease, marriage status,
  work type, residence type, glucose level, BMI, and smoking status

### Obesity Effect on Disease Rates

| Metric | Obese (BMI >= 30) | Non-obese (BMI < 30) | Relative Risk |
|--------|-------------------|----------------------|---------------|
| **Stroke Rate** | 5.10% | 4.73% | **1.08x** |
| **Heart Disease Rate** | 6.25% | 4.89% | **1.28x** |

### Stroke Prediction

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|----|-----|
| **Logistic Regression** | 74.1% | 13.6% | 80.0% | 23.2% | **0.839** |
| Random Forest | 92.1% | 19.6% | 20.0% | 19.8% | 0.824 |
| Gradient Boosting | 94.8% | 33.3% | 6.0% | 10.2% | 0.797 |
| XGBoost | 92.5% | 22.4% | 22.0% | 22.2% | 0.812 |
| MLP Neural Net | 95.1% | 0.0% | 0.0% | 0.0% | 0.543 |

**Best Model**: Logistic Regression (AUC = 0.839)

### Heart Disease Prediction

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|----|-----|
| **Logistic Regression** | 75.6% | 14.6% | 72.7% | 24.3% | **0.830** |
| Random Forest | 87.6% | 14.0% | 25.5% | 18.1% | 0.798 |
| Gradient Boosting | 93.4% | 16.7% | 5.5% | 8.2% | 0.785 |
| XGBoost | 89.7% | 16.2% | 21.8% | 18.6% | 0.803 |
| MLP Neural Net | 94.6% | 0.0% | 0.0% | 0.0% | 0.485 |

**Best Model**: Logistic Regression (AUC = 0.830)

## 2. Visualizations Generated

1. **bmi_disease_risk.png** - Stroke and heart disease rates across BMI categories
2. **model_comparison.png** - Classification metrics across all models
3. **feature_importance.png** - XGBoost feature importances for stroke prediction
4. **obesity_effect_summary.png** - Disease rates by obesity status

## 3. Conclusions

1. **Obesity is associated with higher individual disease risk**: obese
   individuals show 1.08x stroke risk and 1.28x heart disease risk in the
   clinical dataset.
2. **Age remains the dominant predictor**: age consistently ranks as the most
   important feature, followed by glucose level and clinical risk indicators.
3. **Simple models perform strongly**: Logistic Regression achieved the best
   AUC for both stroke and heart disease prediction, while the neural network
   struggled with class imbalance.
