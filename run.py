import os
import subprocess
from pathlib import Path

if __name__ == "__main__":
    project_root = Path(__file__).parent.absolute()
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    ui_script = project_root / "app" / "ui.py"

    print(f"[Core] Initializing Hybrid-RAG Instance from: {project_root}")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root)

    subprocess.run(
        [str(venv_python), "-m", "streamlit", "run", str(ui_script), "--server.port=8501"],
        env=env,
    )