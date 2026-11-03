"""Join discovery evidence without collapsing models into weight identities.

The result is a migration worklist, not an ongoing catalog. Model cards and
benchmark eligibility remain incomplete until source and scientific review.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

from discover import DECLARATIONS, protected


ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def receipt_groups(local):
    grouped = {}
    for row in local['receipts']:
        if row.get('training_receipt_candidate') or row['read_status'] == 'unsupported_non_object_receipt':
            grouped.setdefault(row['sha256'], []).append(row)
    training, other = [], []
    for identity, rows in sorted(grouped.items()):
        rows.sort(key=lambda r: ('dependency-migration-v1/' in r['path'], len(r['path']), r['path']))
        path = ROOT / rows[0]['path']
        if sha(path) != identity:
            raise ValueError('Receipt changed after discovery')
        value = json.loads(path.read_text())
        paths = [r['path'] for r in rows]
        if isinstance(value, dict):
            architecture = value.get('architecture', {})
            training.append(dict(receipt_sha256=identity, paths=paths,
                schema=value.get('schema'), mode=value.get('mode'),
                optimizer_updates=value.get('optimizer_updates'), parameter_fits=value.get('parameter_fits'),
                architecture=architecture, parameter_count=value.get('parameter_count'),
                outputs=value.get('outputs'), closure=value.get('closure'),
                interpretation='Training receipt candidate; confirm fit versus recovery and exact data scope.'))
        else:
            fixture = (isinstance(value, list) and bool(value) and all(isinstance(v, dict)
                       and {'fixture', 'command', 'byte_identical_sha256', 'equal_manifest_fields'} <= set(v)
                       for v in value))
            other.append(dict(receipt_sha256=identity, paths=paths,
                              classification='native_fixture_replay_verification' if fixture else 'unresolved_nonobject',
                              entries=len(value) if isinstance(value, list) else None,
                              model_training_receipt=False if fixture else None))
    return training, other


def prototype_sources():
    result = subprocess.run(['rg', '--files', '--null', '-g', '*.py', 'experiments'],
                            cwd=ROOT, capture_output=True, check=True)
    rows, errors = [], []
    for name in sorted(p.decode() for p in result.stdout.split(b'\0') if p):
        if protected(Path(name)) or name.startswith('experiments/model-portfolio-migration-v1/'):
            continue
        path = ROOT / name
        try:
            tree = ast.parse(path.read_text(), filename=name)
        except (SyntaxError, UnicodeError, OSError) as error:
            errors.append(dict(path=name, error_type=type(error).__name__))
            continue
        classes, neural_calls = [], set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = [ast.unparse(b) for b in node.bases]
                if (any(b.endswith('.Module') for b in bases)
                        or re.search(r'Pressure|Model|Generator|Sampler|Graph', node.name)):
                    classes.append(dict(name=node.name, bases=bases, line=node.lineno))
            elif isinstance(node, ast.Call):
                name_called = ast.unparse(node.func)
                if re.fullmatch(r'(torch\.)?nn\.(Linear|Conv[123]d|MultiheadAttention|Transformer.*|GRU.*|LSTM.*)', name_called):
                    neural_calls.add(name_called)
        if classes or neural_calls or path.name in ('model.py', 'architecture.py'):
            rows.append(dict(path=name, sha256=sha(path), classes=classes,
                             neural_constructors=sorted(neural_calls),
                             disposition='source_candidate_requires_model_or_helper_classification'))
    return rows, errors


def choose_checkpoint(record, paths):
    if isinstance(record, dict):
        artifacts = record.get('replay_artifacts')
        if isinstance(artifacts, str):
            selected = (ROOT / 'docs/plans' / artifacts / 'weights.npz').resolve()
            if selected.is_relative_to(ROOT):
                name = str(selected.relative_to(ROOT))
                if name in paths:
                    return name
    return paths[0]


def reconcile(local_path, weights_path, catalog_path, source_bindings_path, output, summary_path):
    if output.exists() or summary_path.exists():
        raise ValueError('Fresh reconciliation outputs required')
    local, weights, catalog = (json.loads(p.read_text()) for p in (local_path, weights_path, catalog_path))
    source_bindings = json.loads(source_bindings_path.read_text())
    if (source_bindings['status'] != 'verified'
            or source_bindings['observation']['sha256'] != sha(catalog_path)):
        raise ValueError('Exact verified retained source observation required')
    retained_sources = {m['ref']['sha256']: m for m in source_bindings['models']}
    available = {g['sha256']: g for g in weights['groups']}
    dashboard_path = ROOT / local['dashboard']['path']
    if sha(dashboard_path) != local['dashboard']['sha256']:
        raise ValueError('Dashboard changed after discovery')
    raw = dashboard_path.read_bytes()
    dashboard = json.loads(re.search(rb'<script id="chart-data" type="application/json">(.*?)</script>', raw, re.S)[1])
    declarations = []
    for row in local['declarations']:
        scope, index = row['discovery_key'].rsplit('/', 1)
        value = dashboard
        for name in DECLARATIONS[scope]:
            value = value[name]
        record = value[int(index)]
        identity = row['metadata'].get('weights_sha256')
        weight = available.get(identity)
        declarations.append(dict(**row, record=record,
            checkpoint_disposition=('available_content_verified' if weight else
                                    'declared_content_not_located' if identity else
                                    'no_own_checkpoint_declared_resolve_components_or_reference'),
            checkpoint_path=(choose_checkpoint(record, weight['paths']) if weight else None),
            checkpoint_copy_count=(len(weight['paths']) if weight else 0)))
    training, other = receipt_groups(local)
    sources, errors = prototype_sources()
    models = catalog['captured_metadata']['models']
    material_by_id = {r['material']['model_definition_id']: r for r in catalog['captured_metadata']['model_materials']}
    catalog_models = []
    for model in models:
        material = material_by_id.get(model['id'])
        source_records = []
        retained = retained_sources.get(material['ref']['sha256']) if material else None
        retained_by_path = {r['import_path']: r for r in retained['sources']} if retained else {}
        if material:
            for row in material['material']['source_closure']:
                relative = Path(row['import_path'])
                if relative.is_absolute() or '..' in relative.parts or protected(relative):
                    source_records.append(dict(path=str(relative), status='requires_separate_path_review'))
                    continue
                paths = [ROOT / relative, ROOT / 'runs/dependency-migration-v1/lab' / relative]
                matches = [str(p.relative_to(ROOT)) for p in paths
                           if p.is_file() and not p.is_symlink() and sha(p) == row['blob']['sha256']]
                status = 'available_exact_source' if matches else 'exact_source_not_located_at_declared_paths'
                if not matches and str(relative) in retained_by_path:
                    saved = retained_by_path[str(relative)]
                    if saved['blob'] != row['blob']:
                        raise ValueError('Retained source proof differs from captured inventory')
                    matches = [saved['retained_path']]
                    status = 'available_exact_retained_source'
                source_records.append(dict(path=str(relative), expected_sha256=row['blob']['sha256'],
                                           matches=matches, status=status))
        catalog_models.append(dict(id=model['id'], name=model['name'], revision=model['revision'],
                                   specification=model['specification'], source_selector=model['source_selector'],
                                   published_material=material['ref'] if material else None,
                                   published_policies=retained['policies'] if retained else None,
                                   source_records=source_records,
                                   disposition='requires_link_to_training_receipt_checkpoint_and_dashboard_aliases'))
    value = dict(schema='rheon.model-portfolio-reconciliation.v1', created_unix=time.time(),
                 status='partial_identity_reconciliation_not_complete_model_inventory',
                 inputs=[dict(path=str(p), sha256=sha(p)) for p in (local_path, weights_path, catalog_path, source_bindings_path)],
                 source_sha256=sha(Path(__file__)), declarations=declarations,
                 training_receipt_groups=training, nonobject_receipt_groups=other,
                 catalog_models=catalog_models, prototype_source_candidates=sources,
                 prototype_source_parse_errors=errors,
                 full_run_collection_captured=catalog['full_run_collection_captured'],
                 remaining=['Reconcile extensionless and externally retained checkpoint artifacts, including historical C01/C02.',
                            'Account for the complete Run collection; the expensive legacy list timed out.',
                            'Classify every source candidate as model, composition, helper, fixture or duplicate source alias.',
                            'Map each catalog definition, training receipt, checkpoint and declaration into model identities.',
                            'Finish method, purpose, lessons and executable reproduction cards before benchmark coverage is complete.'],
                 new_inference_calls=0, complete_model_count_established=False)
    with output.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    declared = {r['metadata'].get('weights_sha256') for r in declarations} - {None}
    summary = dict(schema=value['schema'], status=value['status'], output=str(output), output_sha256=sha(output),
                   declarations=len(declarations), distinct_declared_weight_hashes=len(declared),
                   declared_weight_hashes_content_located=len(declared & available.keys()),
                   declared_weight_hashes_not_located=sorted(declared - available.keys()),
                   checkpoint_byte_identities=len(available), catalog_model_definitions=len(catalog_models),
                   training_receipt_groups=len(training), training_receipt_modes=dict(Counter(r['mode'] for r in training)),
                   other_receipt_classifications=dict(Counter(r['classification'] for r in other)),
                   prototype_source_candidates=len(sources), prototype_source_parse_errors=errors,
                   published_source_entries=dict(Counter(r['status'] for m in catalog_models for r in m['source_records'])),
                   full_run_collection_captured=value['full_run_collection_captured'],
                   new_inference_calls=0, complete_model_count_established=False)
    with summary_path.open('x') as stream:
        json.dump(summary, stream, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-inventory', type=Path, required=True)
    parser.add_argument('--checkpoint-identities', type=Path, required=True)
    parser.add_argument('--catalog-observation', type=Path, required=True)
    parser.add_argument('--catalog-source-bindings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    reconcile(args.local_inventory, args.checkpoint_identities, args.catalog_observation,
              args.catalog_source_bindings, args.output, args.summary)
