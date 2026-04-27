from __future__ import annotations

import base64
import os
import subprocess
import sys
from pathlib import Path


def _to_data_uri(path: Path) -> str:
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_dashboard(task_dir: Path) -> Path:
    outputs = task_dir / "outputs"
    plots = outputs / "plots"

    report_text = (outputs / "dataset_report.txt").read_text(encoding="utf-8")
    metrics_text = (outputs / "model_metrics.txt").read_text(encoding="utf-8")

    plot_files = [
        ("EDA: Target Distribution", plots / "eda_target_distribution.png"),
        ("EDA: Correlation Heatmap", plots / "eda_correlation_heatmap.png"),
        ("EDA: Age vs Cholesterol", plots / "eda_age_vs_chol.png"),
        ("Model: ROC Curve", plots / "model_roc_curve.png"),
        ("Model: Confusion Matrix", plots / "model_confusion_matrix.png"),
        ("Model: Feature Importance", plots / "model_feature_importance.png"),
    ]

    sections = []
    for title, file_path in plot_files:
        if file_path.exists():
            sections.append(
                f"<div class='card'><h2>{title}</h2><img src='{_to_data_uri(file_path)}' alt='{title}' /></div>"
            )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Task 2 Dashboard - Heart Disease Prediction</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f6f7fb; color: #1f2937; }}
    h1 {{ margin-top: 0; }}
    .card {{ background: white; border: 1px solid #d5d9e0; border-radius: 10px; padding: 16px; margin-bottom: 18px; }}
    pre {{ white-space: pre-wrap; background: #111827; color: #e5e7eb; padding: 12px; border-radius: 8px; overflow-x: auto; }}
    img {{ width: 100%; max-width: 980px; border: 1px solid #d5d9e0; border-radius: 8px; margin-top: 8px; }}
  </style>
</head>
<body>
  <h1>Task 2 - Heart Disease Prediction Dashboard</h1>

  <div class=\"card\">
    <h2>Dataset Report</h2>
    <pre>{report_text}</pre>
  </div>

  <div class=\"card\">
    <h2>Model Metrics</h2>
    <pre>{metrics_text}</pre>
  </div>

  {''.join(sections)}
</body>
</html>
"""

    dashboard = outputs / "dashboard.html"
    dashboard.write_text(html, encoding="utf-8")
    return dashboard


def main() -> int:
    root = Path(__file__).resolve().parent
    task_dir = root / "Task-2-Heart"
    venv_python = task_dir / ".venv" / "Scripts" / "python.exe"
    requirements = task_dir / "requirements.txt"
    script = task_dir / "scripts" / "task2_heart_disease_prediction.py"

    if not task_dir.exists():
        print(f"Task-2 folder not found: {task_dir}")
        return 1

    if not venv_python.exists():
        print("Creating Task-2 virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(task_dir / ".venv")], check=True)
        subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([str(venv_python), "-m", "pip", "install", "-r", str(requirements)], check=True)

    print("Running Task 2...")
    subprocess.run([str(venv_python), str(script)], check=True)

    dashboard = _build_dashboard(task_dir)
    print("Opening dashboard...")
    os.startfile(str(dashboard))  # type: ignore[attr-defined]
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

