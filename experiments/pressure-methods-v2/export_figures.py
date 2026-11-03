"""Use the isolated plotting environment, keeping the qualified training environment unchanged."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
raise SystemExit(subprocess.run(['/tmp/rheon-v2-plot-env/bin/python','-B',str(ROOT/'experiments/pressure-methods-v2/plot_report.py')],cwd=ROOT,timeout=600).returncode)
