"""Freeze a matched-example balanced-cycle Project using existing assembly."""
import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BANK = ROOT / 'experiments/pressure-fundamental-bank-fit-v1'
PREVIOUS = ROOT / 'experiments/pressure-four-pass-distillation-v1'
sys.path.insert(0, str(PREVIOUS))
from common import read, write
from prepare import prepare


def owner():
    spec = importlib.util.spec_from_file_location('bank_preparation', BANK / 'prepare.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(config_path, output):
    config = read(config_path)
    if len(config['arms']) != 1:
        raise ValueError('One bounded aggregate screen required')
    arm = config['arms'][0]
    settings = owner().selected(config, arm)
    count = len(settings['fitting_bank']['grids']) * len(settings['fitting_bank']['modes_zyx'])
    settings.update(updates=config['exposures_per_case'],
        examples=count * config['exposures_per_case'],
        schedule=[dict(update=e + 1, learning_rate=r) for e, r in
                  [(100, 1e-5), (150, 1e-6), (175, 1e-7), (190, 1e-8)]])
    frozen = output.parent / (arm['id'] + '-experiment.json')
    if output.exists() or frozen.exists():
        raise ValueError('Fresh Project and settings required')
    output.parent.mkdir(parents=True, exist_ok=True)
    write(frozen, settings)
    prepare(frozen, arm['id'], output, source_overrides={
        'fine_tuning.py': BANK / 'fine_tuning.py',
        'original_fine_tuning.py': PREVIOUS / 'fine_tuning.py',
        'original_fit_bank.py': BANK / 'fit_bank.py',
        'fit_bank.py': HERE / 'fit_bank.py'},
        objective_overrides={'fitting_bank': settings['fitting_bank']},
        trainer_transform=owner().transform)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.config.resolve(), args.output.resolve())
