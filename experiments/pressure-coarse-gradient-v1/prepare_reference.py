"""Bind a diagnostic to explicit immutable failed and completed captures."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate-project', 'capture-review', 'control-project', 'control-study', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    witness = json.loads((args.candidate_project/'source-witness.json').read_text())
    capture = json.loads(args.capture_review.read_text())
    control = json.loads((args.control_study/'receipt.json').read_text())
    history = json.loads((args.control_study/'train/history.json').read_text())
    config = json.loads((args.candidate_project/'config.json').read_text())
    assert config['training']['updates'] == control['optimizer_updates'] == 5000
    assert capture['status'] == 'passed' and capture['all_shared_numerical_sources_exact']
    assert control['status'] == 'completed' and all(control['artifact_parity'].values())
    assert control['training_receipt']['closure']['inputs'] == witness['inputs']
    assert control['training_receipt']['closure']['normalization_sha256'] == witness['normalization_sha256']
    for name, sha in witness['sources'].items():
        assert digest(args.candidate_project/name) == sha, name
    shared = json.loads((args.control_project/'source-witness.json').read_text())
    control_config = json.loads((args.control_project/'config.json').read_text())
    assert {k:v for k,v in config.items() if k!='id'} == {k:v for k,v in control_config.items() if k!='id'}
    assert all(shared['sources'][name] == sha for name, sha in witness['sources'].items()
               if name not in ('model.py', 'base_model.py', 'coarse.py', 'config.json'))
    result = dict(format='pressure-coarse-gradient-reference-v1',
        configuration_sha256=digest(args.candidate_project/'config.json'), original_sources=witness['sources'],
        sampling_order_sha256=dict(real=control['training_receipt']['closure']['sample_order_sha256'],
            analytical=history['analytical_feedback']['sampling_order_sha256'],
            supplemental=history['supplemental_feedback']['sampling_order_sha256'],
            direction=history['direction_feedback']['sampling_order_sha256'],
            current=history['current_feedback']['sampling_order_sha256']),
        analytical_seeds={str(config['analytical_feedback']['seed']):'analytical',
                          str(config['supplemental_feedback']['seed']):'supplemental'},
        direction_seeds={str(config['direction_feedback']['seed']):'direction',
                         str(config['current_feedback']['seed']):'current'},
        initial_inventory={name:dict(shape=value['shape'],sha256=value['sha256'])
                           for name,value in capture['inventories']['candidate']['inventory'].items()},
        sources={str(path.resolve()):digest(path) for path in [args.capture_review,
            args.candidate_project/'source-witness.json', args.control_study/'receipt.json',
            args.control_study/'train/history.json']},
        scope='Historical full-order hashes plus original source/config identity qualify the replay prefix; no stored original partial gradients exist.')
    args.output.mkdir(parents=True)
    (args.output/'gradient-reference.json').write_text(json.dumps(result, sort_keys=True, indent=2)+'\n')
    shutil.copyfile(Path(__file__).with_name('gradient_screen.py'), args.output/'gradient_screen.py')
    print(json.dumps(dict(status='passed', reference_sha256=digest(args.output/'gradient-reference.json'),
                         original_full_updates=5000, original_streams=5, optimizer_updates=0)))


if __name__ == '__main__':
    main()
