"""Verify actual matched current-feedback weights without asserting accuracy."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / 'pressure-current-policy-v1'),
               str(HERE.parent / 'pressure-throughput-feedback-v1')]
from compare_arms import project
from compare_training import measured


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configuration(reference, candidate, weight):
    expected = json.loads(json.dumps(reference))
    expected['id'] = candidate['id']
    expected['current_feedback']['weight'] = weight
    if candidate != expected or candidate['current_feedback']['stage'] != 'late':
        raise ValueError('only declared current-feedback weight and identity may change')


def validator_cutover(reference, candidate, owner):
    if digest(candidate) != digest(owner):
        raise ValueError('captured custody validator differs from qualified owner')
    def without_monitor(path):
        tree = ast.parse(path.read_text())
        tree.body = [node for node in tree.body
                     if not isinstance(node, ast.FunctionDef) or node.name != '_outputs']
        return ast.dump(tree, include_attributes=False)
    if without_monitor(reference) != without_monitor(candidate):
        raise ValueError('custody change exceeds the qualified live output monitor')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'reference', 'custody-validator', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--candidate', action='append', required=True,
                        help='WEIGHT=completed Study directory')
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if args.output.exists() or len(args.candidate) != 2:
        raise ValueError('fresh report and exactly two declared weights required')
    reference_path = args.reference.resolve(strict=True)
    reference, reference_row = measured(reference_path, root)
    reference_config, reference_witness = project(reference_path, root, reference)
    reference_capture = Path(json.loads((reference_path / 'intent.json').read_text())['assembly']).parent
    reference_history = json.loads((reference_path / 'train/history.json').read_text())
    rows = {}
    sources = []
    for item in args.candidate:
        weight_text, separator, value = item.partition('=')
        weight = float(weight_text)
        if not separator or weight not in (0.01, 0.03) or weight_text in rows:
            raise ValueError('unique declared weights 0.01 and 0.03 required')
        study = Path(value).resolve(strict=True)
        receipt, row = measured(study, root)
        config, witness = project(study, root, receipt)
        configuration(reference_config, config, weight)
        capture = Path(json.loads((study / 'intent.json').read_text())['assembly']).parent
        if set(witness['sources']) != set(reference_witness['sources']):
            raise ValueError('captured source inventory changed')
        for name, identity in reference_witness['sources'].items():
            if name == 'config.json':
                continue
            if name == 'custody-validator.py':
                validator_cutover(reference_capture / name, capture / name,
                                  args.custody_validator.resolve(strict=True))
            elif witness['sources'][name] != identity:
                raise ValueError('numerical source changed: ' + name)
        sources.append({k: v for k, v in witness['sources'].items() if k != 'config.json'})
        closure = receipt['training_receipt']['closure']
        for key in ('inputs', 'normalization_sha256', 'runtime_manifest_sha256', 'sample_order_sha256'):
            if closure[key] != reference['training_receipt']['closure'][key]:
                raise ValueError('matched material changed: ' + key)
        if row['initial_checkpoint_sha256'] != reference_row['initial_checkpoint_sha256']:
            raise ValueError('initial checkpoint differs')
        history = json.loads((study / 'train/history.json').read_text())
        for stream in ('analytical_feedback', 'supplemental_feedback', 'direction_feedback', 'optimizer_work'):
            if history[stream] != reference_history[stream]:
                raise ValueError('sampling or actual optimizer work differs: ' + stream)
        expected_feedback = dict(reference_history['current_feedback'], weight=weight)
        if history['current_feedback'] != expected_feedback:
            raise ValueError('current stream differs beyond selected weight')
        row.update(current_feedback=history['current_feedback'], optimizer_work=history['optimizer_work'])
        rows[weight_text] = row
    if {float(k) for k in rows} != {0.01, 0.03} or sources[0] != sources[1]:
        raise ValueError('matched arm inventory or source differs')
    result = dict(format='pressure_current_weight_matched_v1', status='passed',
                  source_sha256=digest(Path(__file__)), reference=reference_row, candidates=rows,
                  scientific_accuracy_asserted=False, optimizer_updates_executed_by_report=0,
                  scope='Actual CUDA/CPU artifacts; only new current-feedback weight varies. Qualified custody monitor cutover is separate from numerical source. Reference0.1 is reused, not retrained. Accuracy requires independent trajectories.')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: {'optimizer_s': v['optimizer_s'], 'optimizer_work': v['optimizer_work']}
                      for k, v in rows.items()}))

if __name__ == '__main__':
    main()
