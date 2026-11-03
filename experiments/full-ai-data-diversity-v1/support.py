"""Reuse qualified runtime and owned native transport without changing old studies."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DOCS = ROOT / 'docs/plans/full-ai-data-diversity-v1'
BASE = ROOT / 'runs/full-ai-data-diversity-v1'
sys.path.insert(0, str(ROOT / 'experiments/full-ai-tight-targets-v1'))
import profile_frozen_components
from profile_frozen_components import read, write, digest
