"""Capture existing catalog metadata through negotiated read-only Tuldok APIs.

No Run admission, reconciliation, artifact download or catalog migration. All
pages and individual responses remain in ignored storage. Historical source
bindings are used only to speak the existing server's API, never to start it.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
SERVICE = ROOT / 'runs/dependency-migration-v1/Tuldok-large-input-v3'
CLIENT = ROOT / 'runs/dependency-migration-v1/lab/experiments/dependency-migration-v1'
CATALOG_ID = 'a7da4a8de53541a2a0e2aad08005bb71'
READ_OPERATIONS = {
    'models.list', 'studies.list', 'studies.get', 'studies.trial',
    'studies.lineage', 'recipes.list_drafts', 'recipes.material',
    'models.material', 'runs.list', 'projects.list', 'graphs.list',
    'project_snapshots.list',
}


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(output):
    if output.exists():
        raise ValueError('Fresh capture directory required; preserve earlier observations')
    sys.path[:0] = [str(SERVICE), str(CLIENT)]
    import agent_api
    from qualify_typed_pressure import OwnerHTTP
    owner = OwnerHTTP(agent_api, 'http://127.0.0.1:8091', CATALOG_ID,
                      ROOT / 'runs/tuldok-performance-v1/private/owner-session.txt')
    output.mkdir(parents=True)
    calls = output / 'calls'
    calls.mkdir()
    index = 0

    def call(operation, arguments):
        nonlocal index
        if operation not in READ_OPERATIONS:
            raise ValueError('Read-only inventory operation required')
        index += 1
        prefix = calls / ('%04d-%s' % (index, operation))
        write(prefix.with_suffix('.request.json'), dict(operation=operation, arguments=arguments))
        result = owner.call(operation, arguments)
        write(prefix.with_suffix('.response.json'), result)
        return result

    def pages(operation, field):
        rows, cursor, seen = [], None, set()
        while True:
            arguments = dict(limit=20)
            if cursor is not None:
                arguments['cursor'] = cursor
            page = call(operation, arguments)
            rows.extend(page[field])
            cursor = page['next_cursor']
            if cursor is None:
                break
            if cursor in seen:
                raise ValueError('Repeated collection cursor')
            seen.add(cursor)
        identifiers = [r['id'] for r in rows]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError('Collection changed during capture; duplicate identity')
        print(operation, len(rows), flush=True)
        return rows

    models = pages('models.list', 'models')
    studies = pages('studies.list', 'studies')
    recipes = pages('recipes.list_drafts', 'recipes')
    snapshots = pages('project_snapshots.list', 'projects')
    views, trials = [], []
    for study in studies:
        offset, ids, reservations = 0, [], []
        while True:
            page = call('studies.get', dict(study_id=study['id'], revision=study['revision'],
                                           offset=offset, limit=100))
            if page['study']['revision'] != study['revision']:
                raise ValueError('Study changed during capture')
            ids.extend(page['study']['trial_ids'])
            reservations.extend(page['reservations'])
            offset = page['page']['next_offset']
            if offset is None:
                break
        if len(ids) != study['trial_count'] or len(ids) != len(set(ids)):
            raise ValueError('Study Trial membership was not completely captured')
        views.append(dict(study=study, trial_ids=ids, reservations=reservations))
        trials.extend(call('studies.trial', dict(trial_id=trial_id)) for trial_id in ids)
    print('studies.trial', len(trials), flush=True)
    nodes, cursor, revision, seen = [], None, None, set()
    while True:
        args = dict(limit=20)
        if cursor is not None:
            args.update(cursor=cursor, revision=revision)
        page = call('studies.lineage', args)
        revision = page['revision']
        nodes.extend(page['nodes'])
        cursor = page['next_cursor']
        if cursor is None:
            break
        if cursor in seen:
            raise ValueError('Repeated lineage cursor')
        seen.add(cursor)
    print('studies.lineage', len(nodes), flush=True)
    materials = {}
    for trial in trials:
        ref = trial['material']
        key = json.dumps(ref, sort_keys=True)
        if key not in materials:
            materials[key] = call('recipes.material', dict(ref=ref))
    model_materials = {}
    for material in materials.values():
        ref = material['material']['model']
        key = json.dumps(ref, sort_keys=True)
        if key not in model_materials:
            model_materials[key] = call('models.material', dict(ref=ref))
    print('published materials', len(materials), len(model_materials), flush=True)
    print('runs.list: requesting retained Run metadata and artifact status', flush=True)
    runs = call('runs.list', {})
    print('runs.list', len(runs), flush=True)
    result = dict(schema='rheon.catalog-inventory.v1', created_unix=time.time(),
                  catalog_id=CATALOG_ID, models=models, studies=views, trials=trials,
                  recipes=recipes, project_snapshots=snapshots, lineage=nodes,
                  lineage_revision=revision, recipe_materials=list(materials.values()),
                  model_materials=list(model_materials.values()), runs=runs,
                  projects=call('projects.list', {}), graphs=call('graphs.list', {}),
                  read_operations_only=True, inference_calls=0,
                  source_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                                 (Path(__file__).resolve(), SERVICE / 'agent_api.py',
                                  CLIENT / 'qualify_typed_pressure.py')},
                  limitations=['Current metadata observation, not a transactionally consistent backup.',
                               'Published artifact status is owner-reported; local content identity is reconciled separately.',
                               'Catalog templates and smoke fixtures require scope classification.'])
    write(output / 'inventory.json', result)
    summary = dict(schema=result['schema'], created_unix=result['created_unix'],
                   catalog_id=CATALOG_ID, calls=index,
                   counts={k: len(result[k]) for k in ('models', 'studies', 'trials', 'lineage',
                       'recipes', 'recipe_materials', 'model_materials', 'runs', 'projects',
                       'project_snapshots', 'graphs')},
                   run_states=dict(Counter(r['state'] for r in runs)),
                   artifact_statuses=dict(Counter(r.get('artifact_status', {}).get('status', 'not_reported')
                                                  for r in runs)),
                   read_operations_only=True, inference_calls=0,
                   inventory_sha256=sha(output / 'inventory.json'))
    write(output / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    collect(parser.parse_args().output)
