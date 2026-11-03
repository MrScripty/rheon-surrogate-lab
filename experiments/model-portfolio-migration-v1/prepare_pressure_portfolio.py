"""Bind every named pressure cohort to its own retained Project and restore.

Use recorded replay/admission paths, never a same-weight producer guess. This
freezes offline material for adapter qualification; it does not establish that
every model supports the common case population or CUDA execution.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / 'docs/plans'
OTHER = ROOT / 'runs/dependency-migration-v1/lab'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def documented_path(name):
    primary = (DOCS / name).resolve()
    if primary.is_file() or primary.is_dir():
        return primary
    secondary = (OTHER / 'docs/plans' / name).resolve()
    if secondary.is_file() or secondary.is_dir():
        return secondary
    return primary


def resolve(record):
    if record.get('replay_project') and record.get('replay_artifacts'):
        return documented_path(record['replay_project']), documented_path(record['replay_artifacts'])
    timing = documented_path(record['timing_source'])
    value = read(timing)
    if value.get('replay_project') and value.get('replay_artifacts'):
        return documented_path(value['replay_project']), documented_path(value['replay_artifacts'])
    if timing.parent.name == 'artifacts':
        project = timing.parents[3] / 'project'
        choices = [p.parent for p in timing.parents[2].glob('stage-*/artifacts/receipt.json')
                   if read(p).get('mode') == 'restore']
        if len(choices) != 1:
            raise ValueError('Ambiguous retained CPU restore')
        return project, choices[0]
    admission = read(documented_path(value['admission_source']))
    candidates = [e for e in admission['models'] if e['weights_sha256'] == record['weights_sha256']]
    if len(candidates) != 1:
        raise ValueError('Recorded admission does not identify one exact restore')
    entry = candidates[0]
    return ROOT / entry['project'], ROOT / entry['artifacts']


def run(worklist, output, summary):
    if output.exists() or summary.exists():
        raise ValueError('Fresh portfolio material paths required')
    entries = [e for e in read(worklist)['named_entries'] if e['item_id'].startswith('cohorts/')]
    models, dispositions = [], []
    for entry in entries:
        label, record = entry['label'], entry['record']
        disposition = dict(identifier=label, fit_item_id=entry['fit_item_id'],
                           weights_sha256=entry['declared_weights_sha256'])
        if entry['checkpoint_disposition'] == 'declared_content_not_located':
            disposition.update(status='original_checkpoint_unavailable', reason='Retain unresolved C01/C02 availability; no substitute parameters.')
            dispositions.append(disposition)
            continue
        try:
            project, restore = resolve(record)
            if not project.is_relative_to(ROOT) or not restore.is_relative_to(ROOT):
                raise ValueError('Retained source/restore outside the inventoried lab roots')
            paths = [project / 'source-witness.json', project / 'config.json']
            paths += [restore / name for name in ('weights.npz', 'normalization.json', 'receipt.json', 'history.json')]
            identities = {str(p.relative_to(ROOT)): sha(p) for p in paths}
            if sha(restore / 'weights.npz') != record['weights_sha256']:
                raise ValueError('Recorded Project restore has different checkpoint bytes')
            witness = read(project / 'source-witness.json')
            source_bindings = {str((project / name).relative_to(ROOT)): digest
                               for name, digest in witness['sources'].items()}
            for path, digest in source_bindings.items():
                if sha(ROOT / path) != digest:
                    raise ValueError('Captured source witness differs: ' + path)
            receipt = read(restore / 'receipt.json')
            if receipt.get('mode') != 'restore' or receipt.get('optimizer_updates') != 0:
                raise ValueError('Recorded restore is not a zero-update restore receipt')
            models.append(dict(id=label, name=record['name'], weights_sha256=record['weights_sha256'],
                               project=str(project.relative_to(ROOT)), restore=str(restore.relative_to(ROOT)),
                               identities=identities, source_bindings=source_bindings,
                               fit_item_id=entry['fit_item_id'], original_case_count=record['cases']))
            disposition.update(status='bound_original_Project_and_restore_pending_GPU_qualification',
                               project=str(project.relative_to(ROOT)), restore=str(restore.relative_to(ROOT)))
        except (OSError, ValueError, KeyError) as error:
            disposition.update(status='source_restore_binding_requires_review', error_type=type(error).__name__, reason=str(error))
        dispositions.append(disposition)
    if len(dispositions) != 129:
        raise ValueError('Unexpected named cohort denominator; review before proceeding')
    value = dict(schema='rheon.named-pressure-portfolio-material.v1', created_unix=time.time(),
                 root=str(ROOT), models=models, dispositions=dispositions,
                 worklist=dict(path=str(worklist.relative_to(ROOT)), sha256=sha(worklist)),
                 source_sha256=sha(Path(__file__)), GPU_qualification_complete=False,
                 new_parameter_fits=0, inference_calls=0,
                 common_case_applicability='pending_captured_contract_and_GPU_adapter_execution',
                 complete_portfolio_benchmark=False)
    output.write_text(json.dumps(value, indent=2) + '\n')
    compact = {k: v for k, v in value.items() if k not in ('models', 'dispositions')}
    compact.update(output=str(output.relative_to(ROOT)), output_sha256=sha(output),
                   cohorts=len(dispositions), bound_models=len(models),
                   dispositions=dict(Counter(d['status'] for d in dispositions)),
                   unresolved=[d for d in dispositions if d['status'] != 'bound_original_Project_and_restore_pending_GPU_qualification'])
    summary.write_text(json.dumps(compact, indent=2) + '\n')
    print(json.dumps(compact), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('worklist', 'output', 'summary'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.worklist.resolve(), args.output.resolve(), args.summary.resolve())
