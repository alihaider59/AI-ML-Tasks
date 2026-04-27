from pathlib import Path
from io import StringIO

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main() -> None:
    # Set visual style for consistent, readable plots.
    sns.set_theme(style="whitegrid")

    # Create output directory if it does not exist.
    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Iris dataset and convert to a pandas DataFrame.
    iris = sns.load_dataset("iris")
    df = pd.DataFrame(iris)

    # Save all inspection output in a text report and print to console.
    report_path = output_dir.parents[0] / "dataset_report.txt"

    shape_text = f"\n=== DATASET SHAPE ===\n{df.shape}\n"
    columns_text = f"\n=== COLUMN NAMES ===\n{df.columns.tolist()}\n"
    head_text = f"\n=== FIRST 5 ROWS ===\n{df.head().to_string(index=False)}\n"

    info_buffer = StringIO()
    df.info(buf=info_buffer)
    info_text = "\n=== DATASET INFO ===\n" + info_buffer.getvalue() + "\n"

    describe_text = (
        "\n=== SUMMARY STATISTICS ===\n"
        + df.describe(include="all").to_string()
        + "\n"
    )

    report_text = shape_text + columns_text + head_text + info_text + describe_text
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text)

    # Scatter plot: relationship between sepal length and sepal width by species.
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="sepal_length",
        y="sepal_width",
        hue="species",
        style="species",
        s=80,
    )
    plt.title("Iris: Sepal Length vs Sepal Width")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Sepal Width (cm)")
    plt.tight_layout()
    plt.savefig(output_dir / "scatter_sepal_length_vs_width.png", dpi=300)
    plt.close()

    # Histograms: value distribution for all numeric features.
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols].hist(figsize=(10, 8), bins=15, edgecolor="black")
    plt.suptitle("Iris Numeric Feature Distributions", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "histograms_numeric_features.png", dpi=300)
    plt.close()

    # Box plots: outlier detection for each numeric feature grouped by species.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for idx, col in enumerate(numeric_cols):
        sns.boxplot(data=df, x="species", y=col, ax=axes[idx])
        axes[idx].set_title(f"Box Plot of {col}")
        axes[idx].set_xlabel("Species")
        axes[idx].set_ylabel(f"{col} (cm)")

    plt.tight_layout()
    plt.savefig(output_dir / "boxplots_by_species.png", dpi=300)
    plt.close()

    print("Report saved in:", report_path)
    print("Plots saved in:", output_dir)


if __name__ == "__main__":
    main()
