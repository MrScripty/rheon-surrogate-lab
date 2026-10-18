"""Prepare matched hard-bank Projects using the existing typed assembly owner."""
import argparse
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREVIOUS = ROOT/'experiments/pressure-four-pass-distillation-v1'
sys.path.insert(0, str(PREVIOUS))
from common import read
from prepare import prepare


def build(config_path, arm_id, output):
    config = read(config_path)
    arm = next(row for row in config['arms'] if row['id'] == arm_id)
    material = read(ROOT/config['baseline_material'])
    initial = next(row for row in material['models'] if row['id'] == config['train_initialization'])
    captured = read(ROOT/initial['project']/'config.json')
    supplemental = copy.deepcopy(captured['supplemental_feedback'])
    supplemental.update(config['hard_bank'])
    prepare(config_path, arm_id, output,
        source_overrides={'fine_tuning.py': HERE/'fine_tuning.py', 'original_fine_tuning.py': PREVIOUS/'fine_tuning.py'},
        config_overrides={'supplemental_feedback': supplemental},
        objective_overrides={'low_mode_feedback':dict(weight=arm['low_mode_weight'], max_mode=3, signal_floor_fraction=.1)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.config.resolve(), args.arm, args.output.resolve())
