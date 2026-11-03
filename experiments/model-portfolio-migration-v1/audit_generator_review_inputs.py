"""Bind pending full-generator reviews to their retained source/parameter files.

Availability only: no architectural interpretation, model execution, numerical
equivalence, training-data rehash or full benchmark claim is made here.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
OUTPUT = DOC/'reports/full-generator-review-input-availability-01.json'


def read(path):
    return json.loads(path.read_bytes())


def binding(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(path=str(path.relative_to(ROOT)),sha256=digest,bytes=path.stat().st_size)


def observe(path, expected=None):
    if not path.is_file():
        return dict(path=str(path.relative_to(ROOT)),status='missing',expected_sha256=expected)
    actual = binding(path)
    return dict(actual,status='exact' if expected is None or actual['sha256'] == expected else 'changed',
                expected_sha256=expected)


def run():
    if OUTPUT.exists():
        raise ValueError('Preserve original input availability report')
    work_path = ROOT/'runs/model-portfolio-migration-v1/discovery/model-worklist-05.json'
    work = read(work_path)
    coverage_path = DOC/'reports/review-task-coverage-77.json'
    pending = {r['fit_item_id'] for r in read(coverage_path)['full_state_generator_source_reviews_pending']}
    rows = []
    for fit in work['fit_cards']:
        if fit['item_id'] not in pending:
            continue
        available = [ROOT/p for p in fit['receipt_paths'] if (ROOT/p).is_file()]
        exact = [p for p in available if binding(p)['sha256'] == fit['receipt_sha256']]
        if not exact:
            rows.append(dict(fit_item_id=fit['item_id'],name=fit['name'],status='receipt_unavailable',
                             receipt_candidates=fit['receipt_paths']))
            continue
        receipt_path = exact[0]
        receipt = read(receipt_path)
        intent_path = receipt_path.parent.parent/'intent.json'
        if not intent_path.is_file():
            rows.append(dict(fit_item_id=fit['item_id'],name=fit['name'],status='project_locator_unavailable',
                             receipt=binding(receipt_path)))
            continue
        intent = read(intent_path)
        project = (ROOT/intent['project']).resolve()
        if not project.is_relative_to(ROOT):
            raise ValueError('Project locator leaves the lab')
        config_path = project/'config.json'
        config = observe(config_path,receipt['closure']['config_sha256'])
        sources = [observe(project/name,sha) for name,sha in sorted(receipt['closure']['sources'].items())]
        weights = [dict(name=name,**observe(receipt_path.parent/name,spec['sha256']))
                   for name,spec in receipt['outputs'].items() if name in ('weights.npz','initial-weights.npz')]
        expected_bytes = {name:spec['bytes'] for name,spec in receipt['outputs'].items() if name in ('weights.npz','initial-weights.npz')}
        for weight in weights:
            if weight['status'] == 'exact' and weight['bytes'] != expected_bytes[weight['name']]:
                weight['status'] = 'changed'
        observations = [config,*sources,*weights]
        rows.append(dict(fit_item_id=fit['item_id'],name=fit['name'],status='available' if all(o['status']=='exact' for o in observations) else 'binding_gap',
            receipt=binding(receipt_path),intent=binding(intent_path),project_path=str(project.relative_to(ROOT)),
            configuration=config,original_configuration=read(config_path) if config['status']=='exact' else None,
            sources=sources,retained_weights=weights,declared_horizons=receipt.get('horizons'),
            declared_parameter_metadata=receipt.get('parameters'),optimizer_updates=receipt.get('optimizer_updates'),
            source_review_complete=False,fresh_common_full_generator_benchmark_complete=False))
    if len(rows) != len(pending) != 0:
        raise ValueError('Incomplete pending generator census')
    result = dict(status='pending_full_generator_source_availability_observed',created_unix=time.time(),
        pending_generator_fits=len(pending),status_counts=dict(Counter(row['status'] for row in rows)),rows=rows,
        evidence=[binding(work_path),binding(coverage_path)],producer=binding(Path(__file__)),
        new_model_forwards=0,new_reference_generation=False,native_mutations=0,payload_copies=0,
        training_data_rehashed=False,source_review_completion_claim=False,full_portfolio_complete=False)
    OUTPUT.write_text(json.dumps(result,indent=2)+'\n')
    print('Pending generator input availability:',result['status_counts'])


if __name__ == '__main__':
    run()
