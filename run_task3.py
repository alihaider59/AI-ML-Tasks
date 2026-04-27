from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    task_dir = root / "Task-3-HealthBot"
    venv_python = task_dir / ".venv" / "Scripts" / "python.exe"
    requirements = task_dir / "requirements.txt"
    script = task_dir / "scripts" / "task3_health_query_chatbot.py"

    if not task_dir.exists():
        print(f"Task-3-HealthBot folder not found: {task_dir}")
        return 1

    if not venv_python.exists():
        print("Creating Task-3-HealthBot virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(task_dir / ".venv")], check=True)
        subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([str(venv_python), "-m", "pip", "install", "-r", str(requirements)], check=True)

    print("Running chatbot self-test...")
    subprocess.run([str(venv_python), str(script), "--self-test"], check=True)

    print("\nTo start chat mode, run:")
    print("python .\\run_task3.py --chat")
    return 0


if __name__ == "__main__":
    if "--chat" in sys.argv:
        root = Path(__file__).resolve().parent
        task_dir = root / "Task-3-HealthBot"
        venv_python = task_dir / ".venv" / "Scripts" / "python.exe"
        script = task_dir / "scripts" / "task3_health_query_chatbot.py"
        raise SystemExit(subprocess.call([str(venv_python), str(script)]))

    raise SystemExit(main())
