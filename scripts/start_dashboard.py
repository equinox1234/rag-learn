"""启动 Dashboard"""
import subprocess
import sys
from pathlib import Path

app_path = Path(__file__).resolve().parents[1] / "src/observability/dashboard/app.py"
subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])