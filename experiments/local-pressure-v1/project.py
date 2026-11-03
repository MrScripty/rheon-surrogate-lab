"""Current water-training material producer; dependency selection owns host facts.

The previous captured CPU assembler is retained only as offline migration
reference in experiments/dependency-migration-v1/dependency-reference-project.py.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dependency-migration-v1'))
from dependency_config import load
from dependency_training import assemble as assemble_current


def assemble(output, dependency_config=None):
    return assemble_current(load(dependency_config), Path(output))
