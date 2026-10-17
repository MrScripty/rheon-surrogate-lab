"""Reuse the custody collector with this screen's stronger admission binding."""
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'pressure-coarse-gradient-v1'))
import collect as diagnostic_collector
import run_diagnostic
# Restore this experiment's path after importing the existing independent owner.
sys.path.insert(0,str(HERE))
from run_stability import validate_admission


if __name__=='__main__':
    # The collector requires only the source identity, not the diagnostic entry name.
    original=run_diagnostic.validate_admission
    run_diagnostic.validate_admission=validate_admission
    try:diagnostic_collector.main()
    finally:run_diagnostic.validate_admission=original
