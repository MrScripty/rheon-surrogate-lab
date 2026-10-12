#!/usr/bin/env python3
"""Launch fluid experiments with Pumas Library's installed PyTorch 2.14 runtime.

No package installation, runtime switching, or Pumas configuration mutation.
Use --pumas-root when the desktop launcher-root configuration is elsewhere.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def resolve_runtime(root=None):
    if root is None:
        config = Path.home() / ".config/pumas-library-electron/launcher-root.json"
        root = Path(json.loads(config.read_text())["launcherRoot"])
    version = (root / ".active-version-torch").read_text().strip()
    if version != "v2.14.0":
        raise RuntimeError(f"Select PyTorch v2.14.0 in Pumas Library first; selected: {version}")
    runtime = root / "torch-versions" / version
    python = runtime / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file() or not (runtime / "runtime.json").is_file():
        raise RuntimeError("Pumas managed runtime installation is incomplete")
    probe = subprocess.check_output([str(python), "-c",
        "import torch,json; print(json.dumps({'torch':torch.__version__,'cuda':torch.cuda.is_available()}))"],
        text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    observed = json.loads(probe)
    if observed["torch"].split("+")[0] != "2.14.0":
        raise RuntimeError(f"Installed package version differs: {observed['torch']}")
    return python, observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pumas-root", type=Path)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    python, observed = resolve_runtime(args.pumas_root)
    print(f"Pumas runtime: {python}; torch={observed['torch']}; CUDA={observed['cuda']}", flush=True)
    if not args.args:
        return
    script = Path(__file__).with_name("trajectory_experiment.py")
    raise SystemExit(subprocess.call([str(python), str(script), *args.args],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}))


if __name__ == "__main__":
    main()
