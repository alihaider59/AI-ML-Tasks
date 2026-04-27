from __future__ import annotations

import base64
import os
import subprocess
import sys
from pathlib import Path


def _image_data_uri(image_path: Path) -> str:
    data = image_path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_dashboard(task_dir: Path) -> Path:
    outputs_dir = task_dir / "outputs"
    plots_dir = outputs_dir / "plots"
    report_file = outputs_dir / "dataset_report.txt"

    scatter = plots_dir / "scatter_sepal_length_vs_width.png"
    hist = plots_dir / "histograms_numeric_features.png"
    box = plots_dir / "boxplots_by_species.png"

    report_text = report_file.read_text(encoding="utf-8") if report_file.exists() else "Report not found."

    html = f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Task 1 Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f7f7f9; color: #222; }}
    .card {{ background: #fff; border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin-bottom: 18px; }}
    h1 {{ margin-top: 0; }}
    img {{ width: 100%; max-width: 980px; border: 1px solid #ddd; border-radius: 8px; display: block; margin-top: 10px; }}
    pre {{ white-space: pre-wrap; background: #111; color: #e9e9e9; padding: 12px; border-radius: 8px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>Task 1 - Iris Dataset Dashboard</h1>

  <div class=\"card\">
    <h2>Dataset Report</h2>
    <pre>{report_text}</pre>
  </div>

  <div class=\"card\">
    <h2>Scatter Plot</h2>
    <img src=\"{_image_data_uri(scatter)}\" alt=\"Scatter plot\" />
  </div>

  <div class=\"card\">
    <h2>Histograms</h2>
    <img src=\"{_image_data_uri(hist)}\" alt=\"Histograms\" />
  </div>

  <div class=\"card\">
    <h2>Box Plots</h2>
    <img src=\"{_image_data_uri(box)}\" alt=\"Box plots\" />
  </div>
</body>
</html>
"""

    dashboard_file = outputs_dir / "dashboard.html"
    dashboard_file.write_text(html, encoding="utf-8")
    return dashboard_file


def main() -> int:
    root_dir = Path(__file__).resolve().parent
    task_dir = root_dir / "Task-1-Iris"
    venv_python = task_dir / ".venv" / "Scripts" / "python.exe"
    requirements = task_dir / "requirements.txt"
    script_file = task_dir / "scripts" / "task1_iris_analysis.py"

    if not task_dir.exists():
        print(f"Task-1-Iris folder not found at: {task_dir}")
        return 1

    if not venv_python.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(task_dir / ".venv")], check=True)
        subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([str(venv_python), "-m", "pip", "install", "-r", str(requirements)], check=True)

    print("Running Task 1...")
    subprocess.run([str(venv_python), str(script_file)], check=True)

    dashboard = _build_dashboard(task_dir)
    print("Opening dashboard...")
    os.startfile(str(dashboard))  # type: ignore[attr-defined]
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
