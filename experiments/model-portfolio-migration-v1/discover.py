"""Inventory retained model declarations and local artifacts without inference.

This is a discovery report, not a model registry or scientific validator. Keep
overlapping declarations until their identities are reconciled. Do not open
weights, simulations, private directories or reserved/holdout payloads. The
native Tuldok catalog is an additional required discovery source.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
RECEIPT_MAX_BYTES = 16 * 1024 * 1024
DECLARATIONS = {
    'cohorts': ('cohort_points',),
    'legacy-models': ('models',),
    'pressure-models': ('full_ai_study', 'models'),
    'temporal-models': ('full_ai_study', 'temporal_models'),
    'temporal-executions': ('full_ai_study', 'temporal_executions'),
    'tight-components': ('full_ai_study', 'tight_components'),
    'causal-components': ('full_ai_study', 'causal_components'),
    'controls': ('full_ai_study', 'controls'),
    'expanded-variants': ('full_ai_study', 'expanded_models', 'variants'),
    'residual-fits': ('full_ai_study', 'expanded_models', 'residual_fits'),
    'pilot-captures': ('full_ai_study', 'expanded_models', 'temporal_pilots', 'captures'),
}
SCALAR_FIELDS = (
    'id', 'identifier', 'label', 'name', 'kind', 'record_kind', 'device',
    'weights_sha256', 'parent_label', 'parent_weights_sha256',
    'pressure_component_weights_sha256', 'architecture', 'training_scope',
    'evaluation_source', 'source', 'optimizer_updates', 'neural_parameters',
    'parameter_count', 'horizons', 'status',
)
EXCLUDED_TREES = (
    '.git', '__pycache__', 'site-packages', 'target', 'node_modules', 'sources',
    'Tuldok*', 'Rheon', 'app-env', '.venv', 'private',
)
CHECKPOINT_SUFFIXES = {'.npz', '.pt', '.pth', '.safetensors', '.onnx', '.pkl'}
CHECKPOINT_NAME = re.compile(r'weight|checkpoint|model|state_dict|parameter', re.I)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def protected(path):
    return any('reserved' in part.lower() or 'holdout' in part.lower()
               or part.lower() == 'private' for part in path.parts)


def relative(path):
    return str(path.relative_to(ROOT))


def files(patterns):
    args = ['rg', '--files', '--hidden', '--no-ignore', '--null']
    for pattern in patterns:
        args += ['-g', pattern]
    for name in EXCLUDED_TREES:
        args += ['-g', '!**/' + name + '/**']
    args.append(str(ROOT / 'runs'))
    result = subprocess.run(args, capture_output=True, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError('Local artifact discovery failed: ' + result.stderr.decode(errors='replace'))
    return sorted(Path(name.decode()) for name in result.stdout.split(b'\0') if name)


def dashboard_declarations():
    path = ROOT / 'docs/plans/cohort-progress.html'
    raw = path.read_bytes()
    match = re.search(rb'<script id="chart-data" type="application/json">(.*?)</script>', raw, re.S)
    if match is None:
        raise ValueError('Dashboard has no retained chart-data record')
    data = json.loads(match[1])
    records = []
    for scope, parts in DECLARATIONS.items():
        values = data
        for part in parts:
            values = values[part]
        if not isinstance(values, list) or any(not isinstance(v, (dict, str)) for v in values):
            raise ValueError('Unexpected declaration representation: ' + scope)
        for index, value in enumerate(values):
            if isinstance(value, str):
                metadata = dict(referenced_identity=value)
            else:
                metadata = {n: value[n] for n in SCALAR_FIELDS
                            if n in value and (value[n] is None or isinstance(value[n], (str, int, float, bool)))}
            records.append(dict(
                discovery_key=scope + '/' + str(index),
                source_pointer='/' + '/'.join(parts) + '/' + str(index),
                declaration_form='reference' if isinstance(value, str) else 'record',
                declaration_sha256=sha(json.dumps(value, sort_keys=True, allow_nan=False).encode()),
                metadata=metadata, identity_reconciled=False,
            ))
    return dict(path=relative(path), bytes=len(raw), sha256=sha(raw)), records


def receipt_inventory():
    rows, excluded = [], []
    for path in files(['receipt.json']):
        if protected(path):
            excluded.append(dict(path=relative(path), reason='protected_reserved_or_private_path_not_opened'))
            continue
        size = path.stat().st_size
        row = dict(path=relative(path), bytes=size)
        if size > RECEIPT_MAX_BYTES:
            row['read_status'] = 'unavailable_oversize_receipt_requires_review'
            rows.append(row)
            continue
        raw = path.read_bytes()
        row['sha256'] = sha(raw)
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            row.update(read_status='invalid_json', diagnostic=str(error))
            rows.append(row)
            continue
        if not isinstance(value, dict):
            row['read_status'] = 'unsupported_non_object_receipt'
            rows.append(row)
            continue
        for name in ('schema', 'mode', 'status', 'model_family', 'optimizer_updates', 'parameter_fits'):
            if isinstance(value.get(name), (str, int, bool)):
                row[name] = value[name]
        row.update(read_status='metadata_read',
                   training_receipt_candidate=value.get('mode') == 'train' or path.parent.name == 'train')
        outputs = value.get('outputs')
        if isinstance(outputs, dict):
            row['declared_output_names'] = sorted(outputs)
        rows.append(row)
    return rows, excluded


def checkpoint_inventory():
    rows, excluded = [], []
    for path in files(['*' + suffix for suffix in sorted(CHECKPOINT_SUFFIXES)]):
        if protected(path):
            excluded.append(dict(path=relative(path), reason='protected_reserved_or_private_path_not_opened'))
        elif CHECKPOINT_NAME.search(path.name):
            rows.append(dict(path=relative(path), bytes=path.stat().st_size,
                role='checkpoint_filename_candidate', content_identity_verified=False,
                contents_opened=False))
    return rows, excluded


def tracked_payload_candidates():
    result = subprocess.run(['git', '-C', str(ROOT), 'ls-files', '-z'], capture_output=True, check=True)
    suffixes = CHECKPOINT_SUFFIXES | {'.npy', '.so', '.dll', '.exe', '.bin', '.h5', '.hdf5'}
    return [name.decode() for name in result.stdout.split(b'\0') if name
            and Path(name.decode()).suffix.lower() in suffixes]


def run(output):
    if output.exists():
        raise ValueError('Fresh discovery output required; preserve earlier snapshots')
    began = time.monotonic()
    dashboard, declarations = dashboard_declarations()
    receipts, receipt_exclusions = receipt_inventory()
    checkpoints, checkpoint_exclusions = checkpoint_inventory()
    result = dict(
        format='rheon_model_portfolio_discovery_v1',
        status='completed_local_discovery_identity_reconciliation_pending',
        created_unix=time.time(), source_sha256=sha(Path(__file__).read_bytes()),
        dashboard=dashboard, declarations=declarations,
        declaration_counts=dict(Counter(r['discovery_key'].split('/')[0] for r in declarations)),
        receipts=receipts, checkpoint_filename_candidates=checkpoints,
        protected_paths_not_opened=receipt_exclusions + checkpoint_exclusions,
        tracked_payload_extension_candidates=tracked_payload_candidates(),
        exclusions=dict(source_only_and_dependency_trees=EXCLUDED_TREES,
                        receipt_metadata_size_limit_bytes=RECEIPT_MAX_BYTES),
        remaining=['Read live Tuldok model/Study/Trial/Run and artifact metadata.',
                   'Reconcile checkpoint identities and source-only/unfitted prototypes.',
                   'Resolve unsupported, invalid or oversize receipts and all missing artifacts.',
                   'Construct canonical model cards and the complete applicable benchmark matrix.'],
        complete_model_count_established=False, native_project_created=False,
        new_inference_calls=0, new_parameter_fits=0, payloads_opened=0,
        elapsed_seconds=time.monotonic() - began,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as target:
        json.dump(result, target, indent=2, allow_nan=False)
        target.write('\n')
    print(json.dumps(dict(status=result['status'], declaration_counts=result['declaration_counts'],
        receipts=len(receipts), training_receipt_candidates=sum(r.get('training_receipt_candidate', False) for r in receipts),
        checkpoint_filename_candidates=len(checkpoints), tracked_payload_extension_candidates=len(result['tracked_payload_extension_candidates']),
        elapsed_seconds=result['elapsed_seconds'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args().output)
