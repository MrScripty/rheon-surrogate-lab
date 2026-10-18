"""Use the existing visible managed lifecycle after aggregate qualification."""
import argparse
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'assembly', 'qualification', 'session-file', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    for name, value in vars(args).items():
        setattr(args, name, value.resolve())
    spec = importlib.util.spec_from_file_location('managed_bank_controller',
        ROOT/'experiments/pressure-single-mode-fit-v1/run.py')
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    owner.run(args, qualification_key='balanced_cycle')
