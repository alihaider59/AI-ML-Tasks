# Task 2: Heart Disease Prediction

This task builds a binary classification model to predict heart disease risk using health data.

## Objectives Covered

- Load and clean the heart disease dataset
- Handle missing values
- Perform EDA to understand data trends
- Train a Logistic Regression classifier
- Evaluate with accuracy, ROC curve, and confusion matrix
- Highlight important predictive features

## Project Structure

- `scripts/task2_heart_disease_prediction.py` - main ML pipeline
- `data/heart.csv` - dataset (auto-saved after first online download, or place manually)
- `outputs/dataset_report.txt` - EDA summary report
- `outputs/model_metrics.txt` - model performance metrics
- `outputs/feature_importance.csv` - ranked feature importance table
- `outputs/plots/` - EDA and model visualizations
- `requirements.txt` - Python dependencies

## How to Run

### Simplest (single command from project root)

From `AI-ML-Tasks` root:

```powershell
python .\run_task2.py
```

This will run Task 2 and open one dashboard page with report + plots + metrics.

### Manual run

From the `Task-2-Heart` folder:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\task2_heart_disease_prediction.py
```

## Output Files

- `outputs/dataset_report.txt`
- `outputs/model_metrics.txt`
- `outputs/feature_importance.csv`
- `outputs/plots/eda_target_distribution.png`
- `outputs/plots/eda_correlation_heatmap.png`
- `outputs/plots/eda_age_vs_chol.png` (if columns exist)
- `outputs/plots/model_roc_curve.png`
- `outputs/plots/model_confusion_matrix.png`
- `outputs/plots/model_feature_importance.png`
