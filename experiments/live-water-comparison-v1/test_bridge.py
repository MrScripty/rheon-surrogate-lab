"""Exercise the same current boundary through the live comparison test selector."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dependency-migration-v1'))
from test_model_adapter import RequestTests
