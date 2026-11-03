"""Capture nine managed Projects using one numerical template and exact initial weights."""
import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREVIOUS = ROOT / 'experiments/pressure-four-pass-distillation-v1'
sys.path.insert(0, str(PREVIOUS))
from common import read, sha, write
from prepare import prepare


def build(identifier, mode, output):
    manifest = read(ROOT / 'docs/plans/flow-reasoning-assessment/initializations.json')
    if identifier not in ('C53', 'C44', 'C37') or mode not in ('U', 'A', 'D'):
        raise ValueError('Declared initialization and arm required')
    if sha(ROOT / manifest['source_material']) != manifest['source_material_sha256']:
        raise ValueError('Initialization source material changed')
    entry = next(e for e in manifest['initializations'] if e['id'] == identifier)
    for name, expected in entry['identities'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Initialization evidence changed: ' + name)
    config = read(ROOT / 'experiments/pressure-fundamental-bank-fit-v1/config.json')
    spec = importlib.util.spec_from_file_location('fundamental_preparation', ROOT / 'experiments/pressure-fundamental-bank-fit-v1/prepare.py')
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    key = identifier.lower() + '-' + mode.lower()
    arm = dict(id=key, model_id=key, steps=4, teacher_weight=0., grid_indices=list(range(8)), mode_indices=[0])
    settings = owner.selected(config, arm)
    settings.update(schema='rheon-detached-correction-study-v1', baseline_material=manifest['source_material'],
                    baselines=['C53', 'C44', 'C37'], train_initialization=identifier,
                    validation_passes=[1, 2, 4, 8])
    frozen = output.parent / (key + '-experiment.json')
    if output.exists() or (frozen.exists() and read(frozen) != settings):
        raise ValueError('Fresh immutable Project/configuration required')
    output.parent.mkdir(parents=True, exist_ok=True)
    if not frozen.exists():
        write(frozen, settings)
    prepare(frozen, key, output, template_identifier='C53',
        source_overrides={'fine_tuning.py': HERE / 'fine_tuning.py',
            'original_fine_tuning.py': PREVIOUS / 'fine_tuning.py',
            'fit_bank.py': HERE / 'fit_bank.py',
            'original_fit_bank.py': ROOT / 'experiments/pressure-fundamental-bank-fit-v1/fit_bank.py'},
        objective_overrides={'fitting_bank': settings['fitting_bank'],
            'rollout_feedback': dict(mode=mode, weight=.05, depth_seed=2026101807)},
        trainer_transform=owner.transform)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initialization', required=True)
    parser.add_argument('--mode', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    build(args.initialization, args.mode, args.output.resolve())
