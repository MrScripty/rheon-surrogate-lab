"""Locate hash-bound training configurations for the structured-method gaps.

Reads retained graph metadata and local manifests only. A configuration is
recovered only when graph -> recipe -> source bytes -> original receipt hashes
agree. Neither graph names nor current filenames establish a match by themselves.
No model payload is deserialized or executed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'runs/model-portfolio-migration-v1'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_local(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError('Source is not an available local lab file')
    return path, path.read_bytes()


def recover(output, report):
    if output.exists() or report.exists():
        raise ValueError('Fresh recovery outputs required')
    graph_path = BASE / 'model-method-config-recovery-01/private/selected-graphs.json'
    graph_raw = graph_path.read_bytes()
    graphs = json.loads(graph_raw)['rows']
    work_path = BASE / 'discovery/model-worklist-05.json'
    work = json.loads(work_path.read_bytes())
    wanted = {f['item_id']: f for f in work['fit_cards'] if not f['method']}
    projects = {node['parameters']['project_id'] for row in graphs for node in row['document']['nodes']
                if 'project_id' in node.get('parameters', {})}
    candidates = subprocess.check_output(['rg', '--files', '--hidden', '--no-ignore', '-g', 'project.json', 'runs'],
                                         cwd=ROOT, text=True).splitlines()
    manifests = {}
    for relative in candidates:
        path = ROOT / relative
        if path.stat().st_size > 2 * 1024 * 1024:
            continue
        try:
            raw = path.read_bytes()
            value = json.loads(raw)
        except (OSError, ValueError):
            continue
        if type(value) is dict and value.get('id') in projects and type(value.get('recipes')) is list:
            manifests.setdefault(value['id'], []).append((path, raw, value))
    output.mkdir(parents=True)
    rows = []
    for identity, fit in wanted.items():
        row = dict(fit_item_id=identity, receipt_sha256=fit['receipt_sha256'], state='unresolved', attempts=[])
        expected = (fit['training_closure'] or {}).get('config_sha256')
        if expected is None:
            row['reason'] = 'Original receipt has no direct configuration digest'
        else:
            for graph in graphs:
                if not any(f['fit_item_id'] == identity for f in graph['fits']):
                    continue
                for node in graph['document']['nodes']:
                    parameters = node.get('parameters', {})
                    if not {'project_id', 'recipe_id', 'expected_sha256'} <= set(parameters):
                        continue
                    for path, raw, manifest in manifests.get(parameters['project_id'], []):
                        for ref in manifest['recipes']:
                            if ref.get('id') != parameters['recipe_id']:
                                continue
                            try:
                                recipe_path, recipe_raw = read_local(ref['path'])
                                if sha(recipe_raw) != ref['sha256'] or ref['sha256'] != parameters['expected_sha256']:
                                    raise ValueError('Current recipe bytes do not match the retained graph binding')
                                recipe = json.loads(recipe_raw)
                                sources = [s for s in recipe['sources'] if s['sha256'] == expected]
                                if not sources:
                                    raise ValueError('Recipe source inventory does not bind the receipt configuration digest')
                                config_path, config_raw = read_local(sources[0]['path'])
                                if sha(config_raw) != expected:
                                    raise ValueError('Current configuration bytes differ from the original receipt digest')
                                config = json.loads(config_raw)
                                if type(config) is not dict:
                                    raise ValueError('Configuration is not an object')
                                target = output / (identity + '.json')
                                if target.exists() and target.read_bytes() != config_raw:
                                    raise ValueError('Different configuration sources matched one receipt')
                                target.write_bytes(config_raw)
                                row.update(state='verified_original_config_bytes', config_sha256=expected,
                                    config_path=str(config_path.relative_to(ROOT)), recovered_path=str(target.relative_to(ROOT)),
                                    graph_id=graph['graph_id'], graph_document_sha256=graph['document_string_sha256'],
                                    project_manifest_path=str(path.relative_to(ROOT)), project_manifest_sha256=sha(raw),
                                    recipe_path=str(recipe_path.relative_to(ROOT)), recipe_sha256=sha(recipe_raw),
                                    configuration_keys=list(config), configuration=config)
                            except (OSError, ValueError, KeyError, TypeError) as error:
                                row['attempts'].append(dict(project_manifest=str(path.relative_to(ROOT)), reason=str(error)))
        if row['state'] == 'unresolved' and 'reason' not in row:
            row['reason'] = 'No complete retained graph/recipe/configuration hash chain found in the searched local manifests'
        rows.append(row)
    result = dict(schema='rheon.model-method-config-recovery.v1',
        status='completed_configuration_binding_audit_not_architecture_or_reproduction_qualification',
        worklist_sha256=sha(work_path.read_bytes()), selected_graphs_sha256=sha(graph_raw),
        runner_sha256=sha(Path(__file__).read_bytes()), structured_method_gaps=len(rows),
        verified=sum(r['state'] == 'verified_original_config_bytes' for r in rows),
        unresolved=sum(r['state'] == 'unresolved' for r in rows), rows=rows,
        new_neural_forwards=0, model_payloads_deserialized=False, source_files_changed=False,
        limitations='Configuration identity only. Architecture semantics, purpose, causal lessons and clean reproduction still need review.')
    report.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'structured_method_gaps', 'verified', 'unresolved')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    recover(args.output.resolve(), args.report.resolve())
