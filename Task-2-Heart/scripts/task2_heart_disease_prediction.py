from __future__ import annotations

from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_URLS = [
    "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv",
    "https://raw.githubusercontent.com/37Degrees/DataSets/master/heart.csv",
]


def load_heart_dataset(data_dir: Path) -> pd.DataFrame:
    csv_path = data_dir / "heart.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)

    for url in DATA_URLS:
        try:
            df = pd.read_csv(url)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
            return df
        except Exception:
            continue

    raise FileNotFoundError(
        "Could not load heart dataset. Place a CSV at Task-2/data/heart.csv."
    )


def normalize_target_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lower_to_actual = {col.lower(): col for col in df.columns}

    if "target" in lower_to_actual:
        target_col = lower_to_actual["target"]
    elif "heartdisease" in lower_to_actual:
        target_col = lower_to_actual["heartdisease"]
    elif "ahd" in lower_to_actual:
        target_col = lower_to_actual["ahd"]
    else:
        raise ValueError("Target column not found. Expected target/HeartDisease/AHD.")

    if target_col != "target":
        df = df.rename(columns={target_col: "target"})

    if df["target"].dtype == "object":
        mapping = {"yes": 1, "no": 0, "present": 1, "absent": 0}
        df["target"] = df["target"].astype(str).str.lower().str.strip().map(mapping)

    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace("?", np.nan)

    for col in df.columns:
        if col == "target":
            continue
        if df[col].dtype == "object":
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() >= int(0.8 * len(df)):
                df[col] = converted

    df = df.dropna(subset=["target"])
    return df


def save_report_text(df: pd.DataFrame, output_dir: Path) -> None:
    report_file = output_dir / "dataset_report.txt"
    info_buffer = StringIO()
    df.info(buf=info_buffer)

    text = (
        "=== DATASET SHAPE ===\n"
        f"{df.shape}\n\n"
        "=== COLUMN NAMES ===\n"
        f"{df.columns.tolist()}\n\n"
        "=== FIRST 5 ROWS ===\n"
        f"{df.head().to_string(index=False)}\n\n"
        "=== DATASET INFO ===\n"
        f"{info_buffer.getvalue()}\n"
        "=== SUMMARY STATISTICS ===\n"
        f"{df.describe(include='all').to_string()}\n"
    )
    report_file.write_text(text, encoding="utf-8")
    print(text)


def run_eda(df: pd.DataFrame, plots_dir: Path) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 4))
    target_counts = df["target"].value_counts().sort_index()
    x_labels = [str(int(x)) for x in target_counts.index]
    sns.barplot(x=x_labels, y=target_counts.values, hue=None)
    plt.title("Target Class Distribution")
    plt.xlabel("Target")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(plots_dir / "eda_target_distribution.png", dpi=300)
    plt.close()

    numeric_df = df.select_dtypes(include="number")
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(plots_dir / "eda_correlation_heatmap.png", dpi=300)
    plt.close()

    if {"age", "chol"}.issubset(df.columns):
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x="age", y="chol", hue="target", alpha=0.8)
        plt.title("Age vs Cholesterol by Heart Disease Target")
        plt.tight_layout()
        plt.savefig(plots_dir / "eda_age_vs_chol.png", dpi=300)
        plt.close()


def train_and_evaluate(df: pd.DataFrame, output_dir: Path, plots_dir: Path) -> None:
    x = df.drop(columns=["target"])
    y = df["target"].astype(int)

    categorical_cols = x.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = x.select_dtypes(include=["number"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)

    metrics_text = (
        "=== MODEL METRICS ===\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"ROC-AUC: {auc:.4f}\n\n"
        "=== CONFUSION MATRIX ===\n"
        f"{cm}\n\n"
        "=== CLASSIFICATION REPORT ===\n"
        f"{clf_report}\n"
    )
    (output_dir / "model_metrics.txt").write_text(metrics_text, encoding="utf-8")
    print(metrics_text)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"Logistic Regression (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(plots_dir / "model_roc_curve.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(plots_dir / "model_confusion_matrix.png", dpi=300)
    plt.close()

    preprocessor_fitted = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_names = preprocessor_fitted.get_feature_names_out()
    coeffs = classifier.coef_[0]

    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coeffs,
                "abs_coefficient": np.abs(coeffs),
            }
        )
        .sort_values("abs_coefficient", ascending=False)
        .head(12)
    )
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, y="feature", x="abs_coefficient", hue=None, orient="h")
    plt.title("Top Important Features (|Logistic Coefficient|)")
    plt.xlabel("Absolute Coefficient")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(plots_dir / "model_feature_importance.png", dpi=300)
    plt.close()


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    output_dir = base_dir / "outputs"
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_heart_dataset(data_dir)
    df = normalize_target_column(df)
    df = clean_dataset(df)

    save_report_text(df, output_dir)
    run_eda(df, plots_dir)
    train_and_evaluate(df, output_dir, plots_dir)

    print(f"\nSaved outputs in: {output_dir}")


if __name__ == "__main__":
    main()
