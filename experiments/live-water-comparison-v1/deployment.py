"""Explicit offline package conversion command for the fixed migration models."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dependency-migration-v1'))
from package_model_adapter import main

if __name__ == '__main__':
    main()
